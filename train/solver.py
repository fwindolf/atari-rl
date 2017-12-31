

class Solver:
    def __init__(self, optimizer, loss, stop_epoch, batchsize=100):
        """
        Create a new solver class
        
        optimizer (torch.optim) : Optimizer for trainig the model
        loss (torch.nn)         : Loss function 
        stop_epch (int)         : 
        """
        
        self.optimizer = optimizer
        self.loss = loss
        self.stop_epch = stop_epoch
        self.batchsize = batchsize
        
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
    
    def train_offline(self, agent, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train only the agents model with the provided data from a data loader.

        Inputs:
        agent (agent)   : The agent with the model that should be trained
        train_loader    : train data in torch.utils.data.DataLoader
        val_loader      : val data in torch.utils.data.DataLoader
        num_epochs (int): total number of training epochs
        log_nth (int)   : log training accuracy and loss every nth iteration
        """        
        
        for epoch in range(num_epochs):
            # Adapted from dl4cv exercise 3 solver
            for i, (inputs, targets) in enumerate(train_loader, 1):
                t_loss = 0 # Loss of this minibatch
                
                inputs, targets = Variable(inputs), Variable(targets)
                if agent.model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda(async=True)

                optim.zero_grad()
                outputs = agent.model(inputs)    
                
                loss = self.loss_func(outputs, targets)
                self.train_loss_history.append(loss.data.cpu().numpy())
                
                t_loss = loss.data.cpu().numpy()[0] # fetch from gpu
                
                loss.backward()                
                optim.step()

                # TODO: Some sort of logging 
                
            _, preds = torch.max(outputs, 1)
            train_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            
            # TODO: Logging
            
    def train_online(self, agent, screen, gamma=0.95, num_epochs=10, log_nth=0):
        """
        Let the agent play in the environment to optimize the strategy
        
        agent (agent)   :
        screen (screen) :
        num_epochs (int): total number of training epochs
        log_nth (int)   : log training accuracy and loss every nth iteration
        """
        
        obs, action, reward, next_obs, done = agent.memory.sample(self.batchsize)
        
        obs_batch = Variable(torch.cat(obs))
        action_batch = Variable(torch.cat(action))
        reward_batch = Variable(torch.cat(reward))
        
        # Observations that dont end the sequence
        non_final_obs = Variable(torch.cat([o for o in next_obs if o is not None])
        
        # Q(s_t, a) -> Q(s_t) from model and select the columns of actions taken
        # .gather() chooses the confidence values that the model predicted at the index of the action
        # that was originally taken
        obs_action_values = model(obs_batch).gather(1, action_batch)
        
        # V(s_t+1) for all next observations
        next_obs_values = Variable(torch.zeros(self.batchsize).type(Tensor))
        next_obs_values[done] = model(non_final_obs).max(1)[0] # future rewards predicted by model
        next_obs_values.volatile = False
        
        expected_obs_action_values = (next_obs_values * gamma) + reward_batch
        
        loss = self.online_loss(obs_action_values, expected_obs_action_values)
                                 
        self.optim.zero_grad()
        loss.backward()
        for param in self.agent.model.parameters():
            param.grad.data.clamp_(-1, 1) # clamp gradient to stay stable
        self.optim.step()        
            
    def play(self, agent, screen):
        """
        
        """
        
        
            
    
    
    