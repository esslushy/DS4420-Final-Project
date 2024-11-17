import json
import numpy as np
import matplotlib.pyplot as plt


def make_graph(gcnn_results, gtransformer_results, empnn_results, x_ticks, x_label, y_label, title, save_loc):
    mean1 = np.mean(gcnn_results, axis=1)
    stddev1 = np.std(gcnn_results, axis=1)

    mean2 = np.mean(gtransformer_results, axis=1)
    stddev2 = np.std(gtransformer_results, axis=1)

    mean3 = np.mean(empnn_results, axis=1)
    stddev3 = np.std(empnn_results, axis=1)

    plt.figure(figsize=(6, 4))

    # Plot each list with error bars
    plt.errorbar(x_ticks, mean1, yerr=stddev1, label='GCNN', fmt='-o', capsize=5)
    plt.errorbar(x_ticks, mean2, yerr=stddev2, label='GTransformer', fmt='-s', capsize=5)
    plt.errorbar(x_ticks, mean3, yerr=stddev3, label='EMPNN', fmt='-^', capsize=5)

    # Add labels and title
    plt.xlabel(x_label)
    plt.xticks(x_ticks)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

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