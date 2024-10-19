import random
from os_ken.base.app_manager import OSKenApp
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.dpid import dpid_to_str
import copy

class SDNQLTRFederatedController(OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SDNQLTRFederatedController, self).__init__(*args, **kwargs)
        self.q_values = {}  # Local Q-values for each switch
        self.global_q_values = {}  # Global Q-values (aggregated)
        self.trust_values = {}  # Trust values for nodes
        self.learning_rate = 0.6
        self.discount_factor = 0.95
        self.aggregation_interval = 10  # Time interval for global aggregation
        print("SDNQLTRFederatedController initialized")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.logger.info("Handshake completed with {}".format(dpid_to_str(datapath.id)))
        self.__add_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt = packet.Packet(msg.data)
        in_port = msg.match['in_port']

        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            return

        src = eth.src
        dst = eth.dst

        # Determine action based on federated learning (using global Q-values)
        action = self.federated_decision(datapath, src, dst)

        actions = [parser.OFPActionOutput(action)]

        # Handle buffer_id correctly
        if msg.buffer_id != ofproto.OFP_NO_BUFFER:
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port, actions=actions)
        else:
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=in_port, actions=actions, data=msg.data)

        self.logger.info(f"Packet from {src} to {dst} forwarded via port {action}")
        datapath.send_msg(out)

        # Update local Q-table and trust values based on actual transmission success
        self.update_local_q_table(datapath.id, src, dst, action, reward=1)  # Reward of 1 for simplicity
        self.update_trust(src, success_rate=0.9)  # Placeholder trust update

        # Periodically aggregate Q-values and update the global model
        if self.should_aggregate_global_model():
            self.aggregate_global_q_values()
            self.redistribute_global_model()

    def federated_decision(self, datapath, src, dst):
        """Make a routing decision based on global Q-values."""
        ofproto = datapath.ofproto
        if src not in self.global_q_values:
            self.global_q_values[src] = {}
        if dst not in self.global_q_values[src]:
            self.global_q_values[src][dst] = {ofproto.OFPP_FLOOD: random.random()}  # Initialize with FLOOD action

        # Make decisions based on the global Q-values (federated learning)
        best_action = max(self.global_q_values[src][dst], key=self.global_q_values[src][dst].get)
        return best_action if self.trust_values.get(src, 1.0) >= 0.5 else ofproto.OFPP_FLOOD  # Fallback to flooding if trust is low

    def update_local_q_table(self, datapath_id, src, dst, action, reward):
        """Update local Q-table using Q-learning."""
        if datapath_id not in self.q_values:
            self.q_values[datapath_id] = {}

        if src not in self.q_values[datapath_id]:
            self.q_values[datapath_id][src] = {}
        if dst not in self.q_values[datapath_id][src]:
            self.q_values[datapath_id][src][dst] = {}

        old_value = self.q_values[datapath_id][src][dst].get(action, 0)
        next_max = max(self.q_values.get(datapath_id, {}).get(dst, {}).values(), default=0)
        self.q_values[datapath_id][src][dst][action] = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.logger.info(f"Updated local Q-value for datapath {datapath_id}, {src}->{dst} action {action}: {self.q_values[datapath_id][src][dst][action]}")

    def aggregate_global_q_values(self):
        """Aggregate Q-values from all local models (switches)."""
        aggregated_q_values = copy.deepcopy(self.q_values)
        for dp_id in self.q_values:
            for src in self.q_values[dp_id]:
                if src not in self.global_q_values:
                    self.global_q_values[src] = {}
                for dst in self.q_values[dp_id][src]:
                    if dst not in self.global_q_values[src]:
                        self.global_q_values[src][dst] = {}

                    # Simple averaging of Q-values for aggregation
                    for action, q_value in self.q_values[dp_id][src][dst].items():
                        if action not in self.global_q_values[src][dst]:
                            self.global_q_values[src][dst][action] = q_value
                        else:
                            self.global_q_values[src][dst][action] = (self.global_q_values[src][dst][action] + q_value) / 2
        self.logger.info("Global Q-values aggregated.")

    def redistribute_global_model(self):
        """Redistribute the aggregated global Q-values to the switches."""
        for dp_id in self.q_values:
            self.q_values[dp_id] = copy.deepcopy(self.global_q_values)
        self.logger.info("Global Q-values redistributed to all switches.")

    def update_trust(self, node, success_rate):
        """Update trust values for a node based on success rate."""
        self.trust_values[node] = 0.9 * self.trust_values.get(node, 1.0) + 0.1 * success_rate
        self.logger.info(f"Updated trust value for node {node}: {self.trust_values[node]}")

    def should_aggregate_global_model(self):
        """Determine if it's time to aggregate the global Q-values."""
        return random.uniform(0, 1) < 0.05  # Example: aggregate 5% of the time, adjust as needed

    def __add_flow(self, datapath, priority, match, actions):
        """Install flow table modification."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        self.logger.info("Flow-Mod written to {}".format(dpid_to_str(datapath.id)))
        datapath.send_msg(mod)
