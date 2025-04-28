import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from gymnasium.spaces.utils import flatten_space
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.torch_utils import FLOAT_MIN
class VPGTorchRLModule(TorchRLModule, ValueFunctionAPI, ):
    """A simple VPG (vanilla policy gradient)-style RLModule for testing purposes.

    Use this as a minimum, bare-bones example implementation of a custom TorchRLModule.
    """
    @override(TorchRLModule)
    def setup(self):
        super().setup()
        input_dim = flatten_space(self.observation_space["observation"]).shape[0]#["observation"]
        hidden_dim_actor = self.model_config["hidden_dim_actor"]
        hidden_dim_critic = self.model_config["hidden_dim_critic"]
        output_dim = 9

        self._policy_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim_actor),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_actor, output_dim),
            #torch.nn.Softmax(dim=-1)  # Use softmax to get a probability distribution over actions
        )
        self._value_function = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim_critic),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_critic, 1)
        )
    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        observation, action_mask = batch[Columns.OBS]["observation"], batch[Columns.OBS]["action_mask"]
        # Convert action mask into an `[0.0][-inf]`-type mask.
        with torch.no_grad():
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            # Get the logits from the policy network.
            logits = self._policy_net(observation)
            # Mask the logits.
            logits_masked = logits + inf_mask
        return {Columns.ACTION_DIST_INPUTS: logits_masked}
    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        observation, action_mask = batch[Columns.OBS]["observation"], batch[Columns.OBS]["action_mask"]
        # Convert action mask into an `[0.0][-inf]`-type mask.
        with torch.no_grad():
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            # Get the logits from the policy network.
            logits = self._policy_net(observation)
            # Mask the logits.
            logits_masked = logits + inf_mask
        return {Columns.ACTION_DIST_INPUTS: logits_masked}
    """
    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        observation, action_mask = batch[Columns.OBS]["observation"], batch[Columns.OBS]["action_mask"]
        # Convert action mask into an `[0.0][-inf]`-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        # Get the logits from the policy network.
        logits = self._policy_net(observation)
        # Mask the logits.
        logits_masked = logits + inf_mask
        
        if len(batch) > 1:
            torch.set_printoptions(profile="full")
            print("logits masked: ", logits_masked)
        return {Columns.ACTION_DIST_INPUTS: logits_masked}
        """
    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        observation, action_mask = batch[Columns.OBS]["observation"], batch[Columns.OBS]["action_mask"]
        # Convert action mask into an `[0.0][-inf]`-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        # Get the logits from the policy network.
        logits = self._policy_net(observation)
        # Mask the logits.
        logits_masked = logits + inf_mask
        return {Columns.ACTION_DIST_INPUTS: logits_masked}
    @override(ValueFunctionAPI)
    def compute_values(self, batch, **kwargs):
        #print("w're in value func !!")
        return self._value_function(batch[Columns.OBS]["observation"]).squeeze(-1)