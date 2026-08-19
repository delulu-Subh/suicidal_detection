from src.train import train_classical
class TrainingPipeline:
 def run(self): return train_classical()
if __name__=='__main__': print(TrainingPipeline().run())
