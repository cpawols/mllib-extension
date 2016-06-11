import pickle

if __name__ == "__main__":
    with open("times.pickle", "rb") as f:
        print pickle.load(file=f)

    with open("scores.pickle", "rb") as f:
        print pickle.load(file=f)
