import random
from os_ken.base.app_manager import OSKenApp
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.dpid import dpid_to_str

class SDNQLTRController(OSKenApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SDNQLTRController, self).__init__(*args, **kwargs)
        self.q_values = {}  # Store Q-values for routing
        self.trust_values = {}  # Trust values for nodes
        self.learning_rate = 0.5
        self.discount_factor = 0.9
        print("SDNQLTRController initialized")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.logger.info("Handshake taken place with {}".format(dpid_to_str(datapath.id)))
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

        # Determine action based on trust values, Q-learning, and link quality
        action = self.trust_based_decision(datapath, src, dst)

        actions = [parser.OFPActionOutput(action)]
        
        # Handle buffer_id correctly
        if msg.buffer_id != ofproto.OFP_NO_BUFFER:
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port, actions=actions)
        else:
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=in_port, actions=actions, data=msg.data)
        
        self.logger.info(f"Packet from {src} to {dst} forwarded via port {action}")
        datapath.send_msg(out)

        # Update Q-table and trust values based on actual transmission success
        # Here you might want to use a more sophisticated success metric based on real transmission feedback.
        self.update_q_table(src, dst, action, reward=1)  # For simplicity, using 1 as reward for now

        # Simulate getting link quality for this transmission
        link_quality = self.get_link_quality(src, dst)  # Placeholder function to get link quality
        self.update_trust_with_link_quality(src, link_quality)

    def trust_based_decision(self, datapath, src, dst):
        ofproto = datapath.ofproto
        if src not in self.q_values:
            self.q_values[src] = {}
        if dst not in self.q_values[src]:
            self.q_values[src][dst] = {ofproto.OFPP_FLOOD: random.random()}  # Initialize with FLOOD action

        # Only choose actions based on trust values and link quality
        best_action = max(self.q_values[src][dst], key=self.q_values[src][dst].get)

        # Consider trust and link quality together in making the decision
        if self.trust_values.get(src, 1.0) >= 0.5 and self.get_link_quality(src, dst) > 70:  # Assuming 70 is good link quality
            return best_action
        else:
            return ofproto.OFPP_FLOOD  # Use flood if trust is low or link quality is poor

    def update_q_table(self, src, dst, action, reward):
        if src not in self.q_values:
            self.q_values[src] = {}
        if dst not in self.q_values[src]:
            self.q_values[src][dst] = {}

        old_value = self.q_values[src][dst].get(action, 0)
        next_max = max(self.q_values.get(dst, {}).get(src, {0: 0}).values())
        self.q_values[src][dst][action] = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.logger.info(f"Updated Q-value for {src}->{dst} action {action}: {self.q_values[src][dst][action]}")

    def update_trust_with_link_quality(self, node, link_quality):
        # Assuming link quality is a value between 0 and 100 (e.g., SNR)
        quality_factor = min(1.0, link_quality / 100)
        # Update trust based on both previous trust and link quality
        self.trust_values[node] = 0.8 * self.trust_values.get(node, 1.0) + 0.2 * quality_factor
        self.logger.info(f"Updated trust for node {node} based on link quality {link_quality}: {self.trust_values[node]}")

    def get_link_quality(self, src, dst):
        """
        Simulates link quality based on the SNR in underwater communication.
        The SNR is affected by distance and environmental conditions.
        """
        
        # Assume some parameters for SNR
        base_snr = 20  # Base SNR in dB at 1m distance
        snr_decay_per_meter = 0.5  # dB loss per meter of distance
        max_distance = 100  # Maximum distance for communication

        # Calculate distance between nodes (you can implement this as needed)
        distance = self.get_distance(src, dst)  # Example function to get distance

        if distance > max_distance:
            return 0  # Quality is zero if distance exceeds max range

        # Calculate SNR based on distance
        snr = base_snr - (snr_decay_per_meter * distance)

        # Convert SNR to a quality metric (0-100 scale)
        quality = max(0, min((snr + 40), 100))  # Assuming SNR of 0 dB corresponds to quality 0
        return quality

    def get_distance(self, src, dst):
        """
        Placeholder function to calculate distance between nodes.
        Replace this with your actual distance calculation logic.
        """
        # Example: Return a fixed distance (this should be replaced with actual distance computation)
        return random.uniform(1, 100)  # Simulating a distance between 1 and 100 meters

    def __add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        self.logger.info("Flow-Mod written to {}".format(dpid_to_str(datapath.id)))
        datapath.send_msg(mod)
