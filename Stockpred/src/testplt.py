

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

student = 'Johnny Doe'
grade = 2
gender = 'boy'
cohortSize = 62  # The number of other 2nd grade boys

numTests = 5
testNames = ['Pacer Test', 'Flexed Arm\n Hang', 'Mile Run', 'Agility',
             'Push Ups']
testMeta = ['laps', 'sec', 'min:sec', 'sec', '']
scores = ['7', '48', '12:52', '17', '14']
rankings = np.round(np.random.uniform(0, 1, numTests)*100, 0)


fig, ax1 = plt.subplots(figsize=(9, 7))
plt.subplots_adjust(left=0.115, right=0.88)
fig.canvas.set_window_title('Feature Importances')
pos = np.arange(numTests) + 0.5  # Center bars on the Y-axis ticks
rects = ax1.barh(pos, rankings, align='center', height=0.5, color='m')

ax1.axis([0, 100, 0, 5])
plt.yticks(pos, testNames)
ax1.set_title('Johnny Doe')
plt.text(50, -0.5, 'Cohort Size: ' + str(cohortSize),
         horizontalalignment='center', size='small')

# Set the right-hand Y-axis ticks and labels and set X-axis tick marks at the
# deciles
ax2 = ax1.twinx()
ax2.plot([100, 100], [0, 5], 'white', alpha=0.1)
ax2.xaxis.set_major_locator(MaxNLocator(11))
xticks = plt.setp(ax2, xticklabels=['0', '10', '20', '30', '40', '50', '60',
                                      '70', '80', '90', '100'])
ax2.xaxis.grid(True, linestyle='--', which='major', color='grey',
               alpha=0.25)
# Plot a solid vertical gridline to highlight the median position
plt.plot([50, 50], [0, 5], 'grey', alpha=0.25)
plt.show()
