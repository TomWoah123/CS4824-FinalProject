from naive_bayes import *
from multilayer_perceptron import *
import matplotlib.pyplot as plt

def main():
  print('Naive Bayes:')
  naive_bayes_acc = naive_bayes_output()
  print('\n-----------------------------------------------')
  print('Multi-layer Perceptron:')
  ml_perceptron_acc = multilayer_perceptron_output()
  print('\n-----------------------------------------------')

  if naive_bayes_acc['test_accuracy'] > ml_perceptron_acc['test_accuracy']:
    print('Naive Bayes performed better with', "{:.5f}".format(naive_bayes_acc['test_accuracy'] * 100) + '% accuracy')
  else:
    print('Multi-layer perceptron performed better with', "{:.5f}".format(ml_perceptron_acc['test_accuracy'] * 100) + '% accuracy')

  # show a graph comparing the two accuracies
  x_axis = [num for num in range(0, 50, 10)]
  mlp_y_axis = [num * 100 for num in ml_perceptron_acc['iteration_accs']]
  nb_y_axis = [naive_bayes_acc['test_accuracy'] * 100 for _ in range(5)]

  plt.plot(x_axis, mlp_y_axis, label="MLP Accuracy")
  plt.plot(x_axis, nb_y_axis, label="Naive Bayes Accuracy")
  plt.title('Accuracies of Naive Bayes vs. Multi-Layer Perceptron')
  plt.xlabel('Iteration')
  plt.ylabel('Accuracy (%)')
  plt.grid(True)
  plt.legend()
  plt.show()

if __name__ == '__main__':
  main()