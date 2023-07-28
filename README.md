# Online Learning in Open Feature and Label Spaces

This paper explores a new problem of Online Learning in Open Feature and Label Spaces, where the dynamic nature
of the feature set demands continuous evolution with the addition
of new features and elimination of outdated ones. This open-ended learning paradigm, unconstrained at both input and output
stages, deals with mostly unlabeled data sequences, potentially
hosting unknown classes. Speciﬁcally, it presents two principal
technical hurdles: i) the deceleration of online convergence and
the disabling of distance measurements due to feature space
dynamics, and ii) the unbounded open-world risk that negatively
interacts with empirical risk during the online process. To
overcome these challenges, we propose a new OOFLS approach,
with its key idea lying in the pursuit of a universal representation
space that delineates a geometric structure underpinning data
sequence. This representation space ensures proximate placement
of data points with probable identical labels and minimizes the
volume of each such known-class region. Our approach buffers
the data points that give rise to tradeoffs between open-world
and empirical risks, enabling optimal rejection to single out low-conﬁdence points from known classes and high-conﬁdence points
from unknown classes. Theoretical results rationalize our universal representation learning design. Extensive experiments across
ﬁve benchmark datasets, against three cutting-edge competitor
models and two ablation variants, substantiate the viability and
superiority of our proposed approach in known-class online
classiﬁcation and unknown-class identiﬁcation.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed Python 3.9.7.
* You have installed PyTorch 1.12.1.

You can check your Python version by running `python --version` in your terminal, and your PyTorch version by running `python -c "import torch; print(torch.__version__)"`.

## Running the Code

We share the code to run Musk Dataset by the following command

```bash
cd source
python train.py
