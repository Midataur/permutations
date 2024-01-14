from utilities import *
from config import *
from dataloading import *
from transformer import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc

# get the model
model = BigramLanguageModel()

# send to gpu (maybe)
if torch.cuda.is_available():
  device = "cuda:0"
else:
  device = "cpu"

# reload the model
filename = PATH + "/model/" + MODELNAME + ".pth"
model.load_state_dict(torch.load(filename, map_location=torch.device(device)))

def generate_predictions(model, dataloader):
   with torch.no_grad():
    model.eval()  # Set the model to evaluation mode

    predictions = []
    for inputs, targets in tqdm(dataloader):
      outputs = model(inputs)
      predictions.append(outputs)

    predictions = torch.cat(predictions)
    predictions = np.array(predictions).reshape(-1)

    return predictions

def create_roc_curve(labels, predictions):
    fpr, tpr, _ = roc_curve(labels, predictions)

    area_under_curve = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve (AUC = {area_under_curve:.2f})")

    # plot the x = y line
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.show()

    return area_under_curve

def create_pr_curve(labels, predictions):
    # plot the precision recall curve
    # read: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    precision, recall, _ = precision_recall_curve(labels, predictions)

    area_under_curve = average_precision_score(labels, predictions)

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve (AUC = {area_under_curve:.2f})')
    plt.show()

    return area_under_curve

def test_suite(dataloader, labels):
    print("Generating predictions...")
    predictions = generate_predictions(model, dataloader)

    print("Creating roc curve...")
    roc_auc = create_roc_curve(labels, predictions)

    print("Creating precision recall curve...")
    pr_auc = create_pr_curve(labels, predictions)

    print("Calculating accuracy...")
    accuracy = accuracy_score(labels, predictions.round())

    print("\nSummary:")
    print("Accuracy:", accuracy)
    print("ROC AUC:", roc_auc)
    print("PR AUC:", pr_auc)

    print("\nIncorrect predictions:")
    for pos, (label, prediction) in enumerate(zip(labels, predictions)):
        if label != prediction.round():
            print(f"Label: {label}, Prediction: {prediction}, index: {pos}")

print("\nTesting artifical data...")
test_suite(art_test_dataloader, y_art_test)

print("\nTesting true data...")
test_suite(true_test_dataloader, y_true_test)
