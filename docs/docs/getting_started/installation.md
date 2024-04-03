## Installation

### Dependencies

To install ``LibMTL``, we recommend to use the following libraries:

- Python == 3.8
- torch == 1.8.1+cu111
- torchvision == 0.9.1+cu111

### User Installation

* Create a virtual environment
  
  ```shell
  conda create -n libmtl python=3.8
  conda activate libmtl
  pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  ```

* Clone the repository
  
  ```shell
  git clone https://github.com/median-research-group/LibMTL.git
  ```

* Install `LibMTL`
  
  ```shell
  cd LibMTL
  pip install -r requirements.txt
  pip install -e .
  ```
