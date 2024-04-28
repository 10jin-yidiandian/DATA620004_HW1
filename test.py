from valid import test_accuracy
import pickle
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_prameters_name = './Mnist_model.pkl'
    f = open(model_prameters_name, 'rb')
    param = pickle.load(f)
    # print(param)
    f.close

    accu = test_accuracy(param)
    print(f'加载模型在测试集上的测试准确率为{accu}')
    print('参数可视化：')
    print(len((param[0]['w']@param[1]['w']).reshape((28,28,10))))
    image_matrices = (param[0]['w']@param[1]['w']).reshape((28,28,10))
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(5, 50))

    # 循环遍历每个矩阵并在相应的子图中显示
    for i in range(10):
        l = i//5
        w = i%5
        axes[l,w].imshow(image_matrices[:,:,i], cmap='gray')
        axes[l,w].axis('off')  # 关闭坐标轴

    # 调整布局以防止重叠
    plt.tight_layout()

    # 保存整个图像
    plt.savefig('combined_images.png')

    # 显示图像
    plt.show()