from modules.protonet import ProtoNet
from modules.protonetloss import PrototypicalLoss
from modules.resnet_aspp import ResNet18ASPPEncoder

encoder = ResNet18ASPPEncoder(use_pretrained=True)
model = ProtoNet(encoder)
criterion = PrototypicalLoss(n_support=5)

