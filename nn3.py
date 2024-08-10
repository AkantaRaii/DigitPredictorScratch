import numpy as np
np.random.seed(0)
# inputs=[1, 2, 3, 2.5]

# weights=[[0.2, 0.8 ,-0.5, 1.0],
#          [0.5, -0.91 ,0.26, -0.5],
#          [-0.26, -0.27 ,0.17, 0.87]]

# biases=[74,3,0.5]
# output=[]
# for weight ,bias in zip(weights,biases):
#     temp=0
#     for w1 ,input in zip(weight,inputs):
#         temp += w1*input
#     temp += bias
#     output.append(temp)
# print(output)
# print(list(zip(inputs,biases)))

inputs=[[1, 2, 3, 2.5],
        [2,  5,   -1,   2],
        [-1.5,2.7,3.3,-0.8]]

weights=[[0.2, 0.8 ,-0.5, 1.0],
         [0.5, -0.91 ,0.26, -0.5],
         [-0.26, -0.27 ,0.17, 0.87]]

biases=[2,3,0.5]

weights2=[[0.1,-0.14,0.5 ],
         [0.5, -0.91 ,0.26, -0.5],
         [-0.26, -0.27 ,0.17, 0.87]]

biases2=[2,3,0.5]


layer1_output=np.dot(inputs,np.array(weights).T)+biases

weights2=[[0.1,-0.14,0.5 ],  #3 ota matra column kina ki layer1_output ma 3 column output aako xa
         [-0.5, 0.12 ,-0.33],
         [-0.44, 0.73 ,-0.13]]

biases2=[-1,2,-0.5]
output=np.dot(layer1_output,np.array(weights2).T)+biases2

print(output)
print(type(output))
