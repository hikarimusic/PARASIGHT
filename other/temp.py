def Conv(c1, c2, k):
    return (k * k * c1 + 2) * c2

def C3(c1, c2, n):
    return int(Conv(c1, c2/2, 1) + Conv(c1, c2/2, 1) + Conv(c2, c2, 1) + n * Conv(c2/2, c2/2, 1) + n * Conv(c2/2, c2/2, 3))
    #return int((5 * n / 2 + 1) * c2 * c2 + (2 * n + 4) * c2 + c1 * c2)

def SPPF(c1, c2, k):
    return int(Conv(c1, c2/2, 1) + Conv(c2*2, c2, 1))

def Detect():
    return (256 + 1) * 48 + (512 + 1) * 48 + (1024 + 1) * 48


print(0, Conv(3, 64, 6))
print(1, Conv(64, 128, 3))
print(2, C3(128, 128, 3))
print(3, Conv(128, 256, 3))
print(4, C3(256, 256, 6))
print(5, Conv(256, 512, 3))
print(6, C3(512, 512, 9))
print(7, Conv(512, 1024, 3))
print(8, C3(1024, 1024, 3))
print(9, SPPF(1024, 1024, 5))
print(10, Conv(1024, 512, 1))
print(11, 0)
print(12, 0)
print(13, C3(1024, 512, 3))
print(14, Conv(512, 256, 1))
print(15, 0)
print(16, 0)
print(17, C3(512, 256, 3))
print(18, Conv(256, 256, 3))
print(19, 0)
print(20, C3(512, 512, 3))
print(21, Conv(512, 512, 3))
print(22, 0)
print(23, C3(1024, 1024, 3))
print(24, Detect())




'''
  0                -1  1      7040  models.common.Conv                      [3, 64, 6, 2, 2]         
  1                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]
  2                -1  3    156928  models.common.C3                        [128, 128, 3]
  3                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]
  4                -1  6   1118208  models.common.C3                        [256, 256, 6]
  5                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]
  6                -1  9   6433792  models.common.C3                        [512, 512, 9]
  7                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 2]
  8                -1  3   9971712  models.common.C3                        [1024, 1024, 3]
  9                -1  1   2624512  models.common.SPPF                      [1024, 1024, 5]
 10                -1  1    525312  models.common.Conv                      [1024, 512, 1, 1]
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 12           [-1, 6]  1         0  models.common.Concat                    [1]
 13                -1  3   2757632  models.common.C3                        [1024, 512, 3, False]
 14                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 16           [-1, 4]  1         0  models.common.Concat                    [1]
 17                -1  3    690688  models.common.C3                        [512, 256, 3, False]
 18                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]
 19          [-1, 14]  1         0  models.common.Concat                    [1]
 20                -1  3   2495488  models.common.C3                        [512, 512, 3, False]
 21                -1  1   2360320  models.common.Conv                      [512, 512, 3, 2]
 22          [-1, 10]  1         0  models.common.Concat                    [1]
 23                -1  3   9971712  models.common.C3                        [1024, 1024, 3, False]
 24      [17, 20, 23]  1     86160  models.yolo.Detect                      [11, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
Model summary: 368 layers, 46192144 parameters, 46192144 gradients, 108.4 GFLOPs
'''