from LibMTL.core import Trainer
    
if __name__ == "__main__":
    Officemodel = Trainer('./office31.yaml')
    Officemodel.train()
