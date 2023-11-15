import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from utils.model_utils import *
from utils.plot_utils import *


def train(X_training, y_training, X_validation, y_validation):

    # Build the neural network model with one hidden layer
    model = keras.Sequential([
        layers.Input(shape=(X_training.shape[1],)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # get summary of model
    model.summary()

    # stochastic gradient descent optimizer
    opt = SGD(learning_rate=0.0001)

    # Compile the model
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # Early Stopping
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=10,
    )

    # Train the model
    history = model.fit(X_training, y_training, epochs=50, batch_size=32, validation_data=(X_validation, y_validation),
                        callbacks=callback, verbose=True)

    return model, history


def plot_figures(model, history, X_testing, y_testing):
    # plot the training accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # plot the training loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_testing, y_testing)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')

    preds = np.round(model.predict(X_testing), 0)

    # confusion matrix
    print(confusion_matrix(y_testing, preds))
    print(classification_report(y_testing, preds))

    # plot ROC curve
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_testing, preds)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    # make the probability predictions 1D
    predictions = preds.flatten()

    # reliability curve
    plot_calibration_curve_nn(predictions, y_test)


if __name__ == '__main__':
    # Load dataset
    df = pd.read_csv('advanced_models_data.csv')

    # Preprocess data
    X_res_scaled, y_res = preprocess_neural_network_corr(df)

    # Split the data into training, validation and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res_scaled, y_res, test_size=0.2, shuffle=True)
    X_train, y_train = balance_data(X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=True)  # 0.25 x 0.8 = 0.2

    # train model
    nn_model, model_history = train(X_train, y_train, X_val, y_val)

    # save model
    nn_model.save("models/neural_network_corr.h5")

    # plot metric figures
    plot_figures(nn_model, model_history, X_test, y_test)

