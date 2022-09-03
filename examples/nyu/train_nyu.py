from LibMTL.core import Trainer
    
if __name__ == "__main__":
    NYUmodel = Trainer('./exp.yaml')
    NYUmodel.train()
