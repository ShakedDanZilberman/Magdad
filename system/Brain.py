from Eye import Eye
from gun import Gun, DummyGun
from cameraIO import Camera


class Brain():
    def __init__(self):
        gun1 = Gun()
        gun2 = Gun()
        eye1 = Eye()
        self.guns = [gun1, gun2]
        self.eyes = [eye1]
        self.targets = []
        self.timestep = 0
    
        
    def get_targets(self):
        pass 
    
    def get_eyes(self):
        pass
    
    def get_guns(self):
        pass
    
    def add(self):
        if self.timestep > 5:
            to_init = False
        else:
            to_init = True
        if self.timestep % 15 == 0:
            to_check = True
        else:
            to_check = False
        
        for eye in self.eyes:
            target = eye.add(to_check, to_init)

    def game_loop(self):
        while True:
            self.timestep += 1
            self.add()


            

        

    
        