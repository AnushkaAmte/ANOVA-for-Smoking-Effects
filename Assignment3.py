from utils import *
import pandas as pd

def main():
    filepath = "../data/Raw Data_GeneSpring.txt"
    data = pd.read_csv(filepath, delimiter = '\t')
    p_values = compute_p(data)
    plot_histogram(p_values)
    get_interesting_rows(data)

if __name__ == "__main__":
    main()

