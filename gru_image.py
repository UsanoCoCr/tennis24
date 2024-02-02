import matplotlib.pyplot as plt
accuracies = [0.5806674020959736, 0.6160529582126604, 0.6504827586206896, 0.6734722030624913, 0.6804635761589404,
              0.6853870567131227, 0.677891250345018, 0.687784679089027, 0.6893981225842076, 0.6961203921027199]
plt.plot(range(1, 11), accuracies)
plt.xlabel('n_steps')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('./image/gru_accuracy_nsteps.png')
plt.show() 