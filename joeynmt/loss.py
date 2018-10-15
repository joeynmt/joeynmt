import torch.nn.functional as F

# TODO replace by nicer loss class, not used right now

class SimpleLossCompute:
    """A simple loss compute function."""

    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, input, target, norm):
        input = F.log_softmax(input, dim=-1)  # log softmax over voc
        #print('log softmax', input.size())  # batch x time x voc
        loss = self.criterion(input=input.contiguous().view(-1, input.size(-1)),
                              target=target.contiguous().view(-1))
        loss = loss / norm

        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()

        # *norm so we can normalize over complete data set later
        return loss.data.item() * norm
