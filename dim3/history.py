import matplotlib.pyplot as plt
import pickle

def show_history_channels(history):
    figure, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

    axs[0].plot(history['dim3_psnr'], label='psnr')

    axs[1].plot(history['val_loss'], label='loss')

    axs[0].legend()
    axs[1].legend()
    plt.show()

# history.pkl 파일을 읽기 모드로 엽니다
with open('../history/history.pkl', 'rb') as f:
    history = pickle.load(f)

print(history.keys())
show_history_channels(history)