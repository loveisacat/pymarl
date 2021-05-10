import torch.nn as nn
import torch.nn.functional as F
import torch

#This is the Dynamic Representation Network(DrNet)
class DrNet(nn.Module):
    def __init__(self, agent_num, output_shape):
        super(DrNet, self).__init__()
        self.enemy_shape = 5
        self.ally_shape = 5
        self.move_direction = 4
        self.unit_shape = 1
        self.n_actions = 6
        self.n_agent = agent_num
        self.n_enemy = agent_num
        self.n_ally = agent_num - 1
        #encoding the enemy observation
        self.fc_enemy = nn.Linear(self.enemy_shape, output_shape)
        #encoding the ally observation
        self.fc_ally = nn.Linear(self.ally_shape, output_shape)


    def forward(self, inputs):
        #The original inputs structure is : move_direction, agent_num * enemy_shape, (agent_num - 1) * ally_shape, unit_shape, n_actions, agent_num
        enemy_inputs = [inputs[:, self.move_direction + i * self.enemy_shape: self.move_direction + (i + 1) * self.enemy_shape] for i in range(self.n_enemy)]
        ally_inputs = [inputs[:, self.move_direction + self.enemy_shape * self.n_enemy + i * self.ally_shape: self.move_direction +  self.enemy_shape * self.n_enemy + (i + 1) * self.ally_shape] for i in range(self.n_ally)]
        own_inputs = torch.cat([inputs[:, :self.move_direction], inputs[:, -(self.n_agent + self.n_actions + self.unit_shape): -self.n_agent]], dim=-1)
        
        enemy_inputs = [F.relu(self.fc_enemy(enemy_inputs[i])) for i in range(self.n_enemy)]
        ally_inputs = [F.relu(self.fc_ally(ally_inputs[i])) for i in range(self.n_ally)]
        
        #For sum aggregate
        x_enemy_sum, x_ally_sum = torch.zeros_like(enemy_inputs[0]), torch.zeros_like(ally_inputs[0])
        for i in range(self.n_enemy):
            x_enemy_sum += enemy_inputs[i]
        for i in range(self.n_ally):
            x_ally_sum += ally_inputs[i]

        x = torch.cat((x_enemy_sum, x_ally_sum, own_inputs), dim=-1)
        return x

if __name__ == '__main__':
    pass
