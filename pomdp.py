import numpy as np
from copy import deepcopy
from Baum_Welch_project import POMDP

class Cheese:
    def __init__(self):
        self.cycles = 200000
        self.gamma = 1.0
        self.horizon = 8
        self.c = 20
        self.T = {}
        self.first_action = 0        

    def cheese(self, s, a):
        r = 0.0
        if s == 10: s_ = np.random.randint(10)
        else:
            s_ = s
        if a == 0:            
            if s == 5: s_ = 0
            if s == 6: s_ = 2
            if s == 7: s_ = 4
            if s == 8: s_ = 5
            if s == 9: s_ = 7
        if a == 1:
            if s == 0: s_ = 5
            if s == 2: s_ = 6
            if s == 4: s_ = 7
            if s == 5: s_ = 8
            if s == 6: s_ = 10
            if s == 7: s_ = 9
        if a == 2 and s > 0 and s < 5 : s_ = s - 1
        if a == 3 and s < 4: s_ = s +1
        if s_ == 0: o = 0
        if (s_ == 1) or (s_ == 3): o = 1
        if s_ == 2: o = 2
        if s_ == 4: o = 3
        if (s_ == 5) or (s_ == 6) or (s_ == 7): o = 4
        if (s_ == 8) or (s_ == 9): o = 5
        if s_ == 10:
            o = 6
            r = 10.0
        return [s_, o, r]
         
    def Rollout(self, s, h, depth):
        if depth == self.horizon: return 0.0
        else:
            a = np.random.randint(4)
            s_, o, r = self.cheese(s, a)            
            return r + self.gamma * self.Rollout(s_, h + "a" + str(a) + "o" + str(o), depth + 1)

    def Search(self, h):
        for _ in range(self.cycles):
            if _ % (self.cycles/100) == 0: print str(int(_/float(self.cycles)*100))+"%"
            if len(h) == 0:
                s = np.random.randint(10)
                self.Simulate(s, h, 0)
            else:
                s = choice(self.T[h][2])
                self.Simulate(s, h, h.count("a"))
            
        tmp = -1
        for b in range(4):
            l = self.T[h + "a" + str(b)][1]
            if l >= tmp:
                tmp = l
                a = b
        self.first_action = a
        
    def Simulate(self, s, h, depth):
        if depth == self.horizon: return 0.0
        if h not in self.T:
            self.T[h] = [1, 0.0, []]
            for a in range(4):
                self.T[h + "a" + str(a)] = [1, 0.0, []]
            return self.Rollout(s, h, depth)
        else:
            tmp = -1            
            for b in range(4):
                l = self.T[h + "a" + str(b)][1] + self.c * np.sqrt(np.log(self.T[h][0])/self.T[h + "a" + str(b)][0])
                if l >= tmp:
                    tmp = l
                    a = b
            s_, o, r = self.cheese(s, a)            
            R = r + self.gamma * self.Simulate(s_, h + "a" + str(a) + "o" + str(o), depth + 1)
            if s not in self.T[h][2]: self.T[h][2].append(s)
            self.T[h][0] += 1
            self.T[h + "a" + str(a)][0] += 1
            self.T[h + "a" + str(a)][1] += (R - self.T[h + "a" + str(a)][1])/ self.T[h + "a" + str(a)][0]            
            return R

    def Policy_Simulation_mean(self, num = 1):        
        mean = 0
        for i in range(num):
            h = ""
            a = self.first_action           
            s = np.random.randint(10)
            states=[s]
            obs = []
            act = [a]            
            total = 0
            for _ in range(self.horizon - 1):
                s, o, r = self.cheese(s,a)
                obs.append(o)                
                total += r
                if num > 1: print "Action:   ", a, "   Observation:  ", o, "   Reward:  ", int(r)
                h += "a" + str(a) + "o" + str(o)
                tmp = -1            
                for b in range(4):
                    l = self.T[h + "a" + str(b)][1]
                    if l >= tmp:
                        tmp = l
                        a = b
                act.append(a)
                states.append(s)
            s, o, r = self.cheese(s,a)
            obs.append(o)
            states.append(s)
            total += r
            if num > 1:
                print "Action:   ", a, "   Observation:  ", o, "   Reward:  ", int(r)
                print 
                print "Total reward:", int(total)
            mean = (1.0/float(i+1))*(i*mean + total)
        if num > 1: return mean
        else: return total, states, obs, act

ch=Cheese()
ch.Search("")
print ch.Policy_Simulation_mean(10)

