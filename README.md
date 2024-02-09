# Utilitarian Online Learning from Open-World Soft Sensing

In Industry 4.0, soft sensing enables to monitor and control com- plex industrial processes in real-time. Whereas recent online al- gorithms bolster predictive modeling on soft sensing streams that increment in volume and vary in feature dimensions, they operate mainly in closed-world settings, where all class labels must be known beforehand. This is restrictive in practical applications like semi- conductor manufacturing, where new wafer defect types emerge dynamically in unforeseeable manners. This study aims to advance online algorithms by allowing learners opt to abstain from make prediction at certain costs. Our key idea is to establish a univer- sal representation space aligning feature dimensions of incoming points while delineating a geometric shape underpinning them. On this shape, we minimize the region spanned by points of known classes through optimizing the tradeoff between empirical risk and abstention cost. Theoretical results rationalize our universal rep- resentation learning design. We benchmark our approach on six datasets, including one real-world dataset of wafer fault-diagnostics collected through chip manufacturing lines in Seagate. Experimen- tal results substantiate the effectiveness of our proposed approach, demonstrating superior performance over four state-of-the-art rival models. Code and datasets are openly accessible via an anonymous link: https://anonymous.4open.science/r/OWSS.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed Python 3.9.7.
* You have installed PyTorch 1.12.1.

You can check your Python version by running `python --version` in your terminal, and your PyTorch version by running `python -c "import torch; print(torch.__version__)"`.

## Soft Sensing Data and Resources

If you would like to better understand the process of handling soft sensing data, 
you can visit the [Soft Sensing Data Repository](https://github.com/Seagate/softsensing_data) on GitHub. 
This repository provides access to a wide range of soft sensing datasets for your research and analysis.

Feel free to explore the datasets available there to gain insights and work with soft sensing data effectively.

## Running the Code

We share the code to run the Seagate Dataset by using the following command:

```bash
cd source
python train.py
