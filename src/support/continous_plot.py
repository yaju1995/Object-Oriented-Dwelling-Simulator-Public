import matplotlib.pyplot as plt


def update_plot(scores):
    """
    Update the dynamic plot with the given scores.

    Parameters:
        scores (list): List of scores to plot.
    """
    plt.clf()  # Clear the current figure
    plt.title("Training Progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    # plt.ylim(-250)  # Set the y-axis limits
    plt.plot(scores, label="Score")  # Plot the scores    plt.text(len(scores) - 1, scores[-1], str(scores[-1]), color="red", fontsize=10)  # Annotate last score
    plt.legend()
    plt.pause(0.1)  # Pause to allow the plot to refresh


def update_two_list_plot(title,list1, list2, label1="List 1", label2="List 2"):
    """
    Update a dynamic plot with two lists.

    Parameters:
        list1 (list): First list of data to plot.
        list2 (list): Second list of data to plot.
        label1 (str): Label for the first list.
        label2 (str): Label for the second list.
    """
    plt.clf()  # Clear the current figure
    plt.title(title)
    plt.xlabel("Number of Entries")
    plt.ylabel("Values")

    # Plot the first list
    plt.plot(list1, label=label1, color="blue")
    if list1:
        plt.text(len(list1) - 1, list1[-1], f"{list1[-1]:.2f}", color="blue", fontsize=10)

    # Plot the second list
    plt.plot(list2, label=label2, color="orange")
    if list2:
        plt.text(len(list2) - 1, list2[-1], f"{list2[-1]:.2f}", color="orange", fontsize=10)

    # Add legend
    plt.legend()
    plt.pause(0.1)  # Pause to allow the plot to refresh

def initialize_plot():
    """
    Initialize the plot for dynamic updating.
    """
    plt.ion()  # Turn on interactive mode


def finalize_plot(show=False):
    """
    Finalize the plot after dynamic updating is complete.
    """
    plt.ioff()  # Turn off interactive mode
    plt.show(block = show)  # Show the final plot