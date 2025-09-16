from sklearn.model_selection import train_test_split

def stratified_split(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
