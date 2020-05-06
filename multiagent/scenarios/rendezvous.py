import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from copy import deepcopy

class Scenario(BaseScenario):
    def make_world(self, na=4, nl=0, random=True):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = na
        num_landmarks = nl
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            # agent.size = 0.15
            agent.size = 0.075
            agent.accel = 1.0
            agent.max_speed = 1.0       
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        if not random:
            if num_agents == 3:
                self.init_pos = np.array([[-0.2, -0.2],[0.2,-0.2],[0, 0.1]])
            elif num_agents == 4:
                self.init_pos = np.array([[0.8,0.8],[-0.8,0.8],[-0.8,-0.8],[0.8,-0.8]])
            else:
                self.init_pos = [np.random.uniform(-1, +1, world.dim_p) for i in range(num_agents)]
            if num_landmarks == 3:
                self.init_landmark = np.array([[-0.7, 0.7],[0.7, 0.7],[0, 0.7]])
            elif num_agents == 4:
                self.init_landmark = np.array([[0, 0.8],[-0.8, 0],[0, -0.8], [0.8, 0]])
            else:
                self.init_landmark = [np.random.uniform(-1, +1, world.dim_p) for i in range(num_agents)]
        self.random = random
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        if self.random:
            self.agent_pos = []
            self.landmark_pos = []
            for agent in world.agents:
                self.agent_pos.append(np.random.uniform(-1, +1, world.dim_p))
                agent.state.p_pos = deepcopy(self.agent_pos[-1])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            for landmark in world.landmarks:
                self.landmark_pos.append(np.random.uniform(-1, +1, world.dim_p))
                landmark.state.p_pos = deepcopy(self.landmark_pos[-1])
                landmark.state.p_vel = np.zeros(world.dim_p)
        else:
            self.agent_pos = []
            self.landmark_pos = []
            # na = float(len(world.agents))
            # nl = float(len(world.landmarks))
            
            for i, agent in enumerate(world.agents):
                # self.agent_pos.append(np.array((0., 2*i/(na-1)-1)))
                self.agent_pos.append(self.init_pos[i])
                agent.state.p_pos = deepcopy(self.agent_pos[-1])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            for i, landmark in enumerate(world.landmarks):
                # self.landmark_pos.append(np.array((2*i/(nl-1)-1, 0.)))
                self.landmark_pos.append(self.init_landmark[i])
                landmark.state.p_pos = deepcopy(self.landmark_pos[-1])
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # individual reward
        rew = -np.linalg.norm(agent.state.p_pos)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
