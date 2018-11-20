from visdom import Visdom
import numpy as np
from matplotlib import pyplot as plt

with open('plate0.out') as log:
    content = log.readlines()
    loss_train = []
    loss_test = []

    num_iter = 0

    for con in content:
        if 'Train Epoch' in con:
            loss_train.append(float(con.split()[6][1:-1]))
            # print(con.split()[5:7])
        if 'Test set' in con:
            loss_test.append(float(con.split()[4][1:-1]))
            # print(con.split()[3:5])

# viz = Visdom(env='loss_curve')
# # print(np.array(loss_train).shape, np.array(range(718)).shape)
# via.text
# viz.line(Y=np.array(loss_train), X=np.array(range(718)), opts=dict(showlegend=True))
# # viz.line(Y=np.array(loss_test), X=np.array(range(718)), opts=dict(showlegend=True))

plt.figure()
plt.plot(np.array(range(718)), np.array(loss_train))
plt.xlabel('Loss')
plt.ylabel('Iters')
plt.title('Loss Curve')
plt.show()
plt.savefig('losscurve.jpg')


Notes:
    1. Supporting multiple logs.
    2. Log file name must end with the lower-cased ".log".
Supported chart types:
    0: Test accuracy  vs. Iters
    1: Test accuracy  vs. Seconds
    2: Test loss  vs. Iters
    3: Test loss  vs. Seconds
    4: Train learning rate  vs. Iters
    5: Train learning rate  vs. Seconds
    6: Train loss  vs. Iters
    7: Train loss  vs. Seconds

I0628 16:51:19.621470 29717 solver.cpp:358] Iteration 0, Testing net (#0)
I0628 16:51:25.335400 29717 solver.cpp:425]     Test net output #0: accuracy_1 = 0.005
I0628 16:51:25.335453 29717 solver.cpp:425]     Test net output #1: accuracy_2 = 0.20125
I0628 16:51:25.335472 29717 solver.cpp:425]     Test net output #2: accuracy_3 = 0.04125
I0628 16:51:25.335489 29717 solver.cpp:425]     Test net output #3: accuracy_4 = 0.195
I0628 16:51:25.335497 29717 solver.cpp:425]     Test net output #4: accuracy_5 = 0.03
I0628 16:51:25.335505 29717 solver.cpp:425]     Test net output #5: accuracy_6 = 0.32875
I0628 16:51:25.335516 29717 solver.cpp:425]     Test net output #6: loss_1 = 5.08069 (* 0.2 = 1.01614 loss)
I0628 16:51:25.335525 29717 solver.cpp:425]     Test net output #7: loss_2 = 1.61186 (* 0.2 = 0.322372 loss)
I0628 16:51:25.335535 29717 solver.cpp:425]     Test net output #8: loss_3 = 2.49053 (* 0.2 = 0.498107 loss)
I0628 16:51:25.335543 29717 solver.cpp:425]     Test net output #9: loss_4 = 2.31941 (* 0.2 = 0.463882 loss)
I0628 16:51:25.335551 29717 solver.cpp:425]     Test net output #10: loss_5 = 2.69787 (* 0.2 = 0.539573 loss)
I0628 16:51:25.335561 29717 solver.cpp:425]     Test net output #11: loss_6 = 1.77341 (* 0.2 = 0.354682 loss)
I0628 16:51:25.513464 29717 solver.cpp:243] Iteration 0, loss = 3.18486
I0628 16:51:25.513496 29717 solver.cpp:259]     Train net output #0: loss_1 = 5.05353 (* 0.2 = 1.01071 loss)
I0628 16:51:25.513507 29717 solver.cpp:259]     Train net output #1: loss_2 = 1.54785 (* 0.2 = 0.309571 loss)
I0628 16:51:25.513519 29717 solver.cpp:259]     Train net output #2: loss_3 = 2.49458 (* 0.2 = 0.498916 loss)
I0628 16:51:25.513527 29717 solver.cpp:259]     Train net output #3: loss_4 = 2.34861 (* 0.2 = 0.469721 loss)
I0628 16:51:25.513537 29717 solver.cpp:259]     Train net output #4: loss_5 = 2.72497 (* 0.2 = 0.544994 loss)
I0628 16:51:25.513547 29717 solver.cpp:259]     Train net output #5: loss_6 = 1.75476 (* 0.2 = 0.350952 loss)
I0628 16:51:25.513568 29717 sgd_solver.cpp:138] Iteration 0, lr = 1e-05