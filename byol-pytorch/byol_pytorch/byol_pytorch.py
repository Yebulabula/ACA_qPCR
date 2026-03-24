import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T
from basenetwork import Transformer_for_byol, Transformer1D

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False, sync_batchnorm = None):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        # print('layer:', layer)
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size, sync_batchnorm = self.sync_batchnorm)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        # print('hidden.shape:', hidden.shape)
        self.hidden.clear()

        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation
        
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        config,
        F_config,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        augment_fn = None,
        augment_fn2 = None,
        moving_average_decay = 0.99,
        use_momentum = True,
        sync_batchnorm = None,
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.hidden_layer = hidden_layer
        self.use_simsiam_mlp = not use_momentum,
        self.sync_batchnorm = sync_batchnorm
        self.config = config
        self.F_config = F_config
        
        # DEFAULT_AUG = torch.nn.Sequential(
        #     RandomApply(
        #         T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        #         p = 0.3
        #     ),
        #     T.RandomGrayscale(p=0.2),
        #     T.RandomHorizontalFlip(),
        #     RandomApply(
        #         T.GaussianBlur((3, 3), (1.0, 2.0)),
        #         p = 0.2
        #     ),
        #     T.RandomResizedCrop((image_size, image_size)),
        #     T.Normalize(
        #         mean=torch.tensor([0.485, 0.456, 0.406]),
        #         std=torch.tensor([0.229, 0.224, 0.225])),
        # )

        # self.augment1 = default(augment_fn, DEFAULT_AUG)
        # self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer = hidden_layer,
            use_simsiam_mlp = not use_momentum,
            sync_batchnorm = sync_batchnorm
        )
        
        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        # self.forward(torch.randn(2, 50, device=device), torch.randn(2, 50, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        device = get_module_device(self.online_encoder)
        target_net = type(self.net)(
            self.config['class_num'],
            self.config['N_seq'],
            **self.F_config,
        ).to(device)
        target_net.load_state_dict(self.net.state_dict())

        target_encoder = NetWrapper(
            target_net,
            self.projection_size,
            self.projection_hidden_size,
            layer = self.hidden_layer,
            use_simsiam_mlp = not self.use_momentum,
            sync_batchnorm = self.sync_batchnorm
        ).to(device)

        # If projector already exists in online encoder, copy its state
        if hasattr(self.online_encoder, 'projector') and self.online_encoder.projector is not None:
            _ = target_encoder._get_projector(torch.zeros_like(
                next(self.online_encoder.projector.parameters())
            ))
            target_encoder.projector.load_state_dict(
                self.online_encoder.projector.state_dict()
            )
        # Freeze the target network
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x,
        y,
        return_embedding = False,
        return_projection = True
    ):  
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(x, return_projection = return_projection)
        
        # Check and ensure both inputs are on the same device
        device = get_module_device(self.online_encoder)
        x, y = x.to(torch.float32).to(device), y.to(torch.float32).to(device)

        image_one, image_two = x, y

        images = torch.cat((image_one, image_two), dim = 0)

        online_projections, _ = self.online_encoder(images)

        online_predictions = self.online_predictor(online_projections)

        online_pred_one, online_pred_two = online_predictions.chunk(2, dim = 0)

        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            target_projections, _ = target_encoder(images)
            target_projections = target_projections.detach()
            target_proj_one, target_proj_two = target_projections.chunk(2, dim = 0)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
