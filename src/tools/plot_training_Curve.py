import pickle
import os
import matplotlib
matplotlib.use('Qt5Agg')  # Use Tkinter backend
import matplotlib.pyplot as plt
# plt.ion()
# import tkinter
# root = tkinter.Tk()
# root.mainloop()

# Define the path to the saved training curve file
dir_out = '7_plex_output'
training_curve_file = os.path.join(dir_out, "training_curve.pkl")

# Load the training_curve dictionary from the saved file
with open(training_curve_file, "rb") as f:
    training_curve = pickle.load(f)

# Extract training and testing accuracy data
train_iters, train_acc = zip(*training_curve["training_accuracy"])
test_iters, test_acc = zip(*training_curve["testing_accuracy"])

# Convert training accuracy to percentages
train_acc_percentage = [x * 100 for x in train_acc]

# Plot training and testing accuracy curves
plt.figure()
plt.plot(train_iters, train_acc_percentage, label="Training Accuracy (%)")
plt.plot(test_iters, test_acc, label="Testing Accuracy (%)")
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.title("Training and Testing Accuracy Curves")
plt.legend()
plt.savefig(os.path.join(dir_out, "training_curve.png"))  # Save plot to file

# plt.show(block=True)
