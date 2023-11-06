# Utilitarian Online Learning from Open-World Soft Sensing

In Industry 4.0, soft sensing enables to monitor and control complex
industrial processes in real-time. Whereas recent online algorithms
bolster predictive modeling on soft sensing streams that increment
in volume and vary in feature dimensions, they operate mainly in
closed-world settings, where all class labels must be known before-
hand. This is restrictive in practical applications like semiconductor
manufacturing, where new wafer defect types emerge dynamically
in unforeseeable manners. This study aims to advance online algo-
rithms by allowing learners opt to abstain from make prediction at
certain costs. Our key idea is to establish a universal representation
space aligning feature dimensions of incoming points while delin-
eating a geometric shape underpinning them. On this shape, we
minimize the region spanned by points of known classes through
optimizing the tradeoff between empirical risk and abstention cost.
Theoretical results rationalize our universal representation learning
design. We benchmark our approach on six datasets, including one
real-world dataset of wafer fault-diagnostics collected through chip
manufacturing lines in Seagate. Experimental results substantiate
the effectiveness of our proposed approach, demonstrating superior
performance over four state-of-the-art rival models.

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
