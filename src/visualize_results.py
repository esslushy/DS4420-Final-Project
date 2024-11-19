import json
import numpy as np
import matplotlib.pyplot as plt


def make_graph(gcnn_results, gtransformer_results, empnn_results, x_ticks, x_label, y_label, title, save_loc):
    mean1 = np.median(gcnn_results, axis=1)
    min1 = np.min(gcnn_results, axis=1)
    max1 = np.max(gcnn_results, axis=1)

    mean2 = np.median(gtransformer_results, axis=1)
    min2 = np.min(gtransformer_results, axis=1)
    max2 = np.max(gtransformer_results, axis=1)

    mean3 = np.median(empnn_results, axis=1)
    min3 = np.min(empnn_results, axis=1)
    max3 = np.max(empnn_results, axis=1)

    plt.figure(figsize=(6, 4))

    # Set the positions and width for the bars
    x = np.arange(len(x_ticks))  # The label locations
    width = 0.25  # The width of the bars

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plotting the bars for each class with error bars
    bars1 = ax.bar(x - width, mean1, width, yerr=[mean1 - min1, max1 - mean1], label='GCNN', capsize=5, color="red")
    bars2 = ax.bar(x, mean2, width, yerr=[mean2 - min2, max2 - mean2], label='GTransformer', capsize=5, color="blue")
    bars3 = ax.bar(x + width, mean3, width, yerr=[mean3 - min3, max3 - mean3], label='EMPNN', capsize=5, color="green")

    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()

    # Show the plot
    plt.savefig(save_loc)

def get_results_starting_with(s, ticks, results, field):
    ordered_results = []
    for t in ticks:
        ordered_results.append(results[s+str(t)][field])
    return ordered_results

def get_ticks_starting_with(s, results):
    ticks_raw = list()
    for k in results.keys():
        if k.startswith(s):
            ticks_raw.append(k)
    return sorted([int(t[len(s):]) for t in ticks_raw])

def main():
    with open("gcnn_test_results.json") as f:
        gcnn_results = json.load(f)
    with open("gtransformer_test_results.json") as f:
        gtransformer_results = json.load(f)
    with open("empnn_test_results.json") as f:
        empnn_results = json.load(f)
    # Get ticks
    equiv_ticks = get_ticks_starting_with("EquivariantWorld", gcnn_results)
    random_ticks = get_ticks_starting_with("RandomWorld", gcnn_results)
    size_ticks = get_ticks_starting_with("SizeWorld", gcnn_results)
    # Get results
    gcnn_equiv_steps = get_results_starting_with("EquivariantWorld", equiv_ticks, gcnn_results, "steps")
    gcnn_equiv_collisions = get_results_starting_with("EquivariantWorld", equiv_ticks, gcnn_results, "collisions")
    gcnn_random_steps = get_results_starting_with("RandomWorld", random_ticks, gcnn_results, "steps")
    gcnn_random_collisions = get_results_starting_with("RandomWorld", random_ticks, gcnn_results, "collisions")
    gcnn_size_steps = get_results_starting_with("SizeWorld", size_ticks, gcnn_results, "steps")
    gcnn_size_collisions = get_results_starting_with("SizeWorld", size_ticks, gcnn_results, "collisions")

    gtransformer_equiv_steps = get_results_starting_with("EquivariantWorld", equiv_ticks, gtransformer_results, "steps")
    gtransformer_equiv_collisions = get_results_starting_with("EquivariantWorld", equiv_ticks, gtransformer_results, "collisions")
    gtransformer_random_steps = get_results_starting_with("RandomWorld", random_ticks, gtransformer_results, "steps")
    gtransformer_random_collisions = get_results_starting_with("RandomWorld", random_ticks, gtransformer_results, "collisions")
    gtransformer_size_steps = get_results_starting_with("SizeWorld", size_ticks, gtransformer_results, "steps")
    gtransformer_size_collisions = get_results_starting_with("SizeWorld", size_ticks, gtransformer_results, "collisions")

    empnn_equiv_steps = get_results_starting_with("EquivariantWorld", equiv_ticks, empnn_results, "steps")
    empnn_equiv_collisions = get_results_starting_with("EquivariantWorld", equiv_ticks, empnn_results, "collisions")
    empnn_random_steps = get_results_starting_with("RandomWorld", random_ticks, empnn_results, "steps")
    empnn_random_collisions = get_results_starting_with("RandomWorld", random_ticks, empnn_results, "collisions")
    empnn_size_steps = get_results_starting_with("SizeWorld", size_ticks, empnn_results, "steps")
    empnn_size_collisions = get_results_starting_with("SizeWorld", size_ticks, empnn_results, "collisions")

    make_graph(gcnn_equiv_steps, gtransformer_equiv_steps, empnn_equiv_steps, equiv_ticks, "Angle", "Steps", "Comparison Between Rotated Worlds", "equiv_steps.pdf")
    make_graph(gcnn_equiv_collisions, gtransformer_equiv_collisions, empnn_equiv_collisions, equiv_ticks, "Angle", "Collisions", "Comparison Between Rotated Worlds", "equiv_collisions.pdf")
    make_graph(gcnn_random_steps, gtransformer_random_steps, empnn_random_steps, random_ticks, "Number of Entities", "Steps", "Comparison Between Worlds with Differing Number of Entities", "random_steps.pdf")
    make_graph(gcnn_random_collisions, gtransformer_random_collisions, empnn_random_collisions, random_ticks, "Number of Entities", "Collisions", "Comparison Between Worlds with Differing Number of Entities", "random_collisions.pdf")
    make_graph(gcnn_size_steps, gtransformer_size_steps, empnn_size_steps, size_ticks, "Size", "Steps", "Comparison Between Worlds of Different Sizes", "size_steps.pdf")
    make_graph(gcnn_size_collisions, gtransformer_size_collisions, empnn_size_collisions, size_ticks, "Size", "Collisions", "Comparison Between Worlds of Different Sizes", "size_collisions.pdf")


if __name__ == "__main__":
    main()