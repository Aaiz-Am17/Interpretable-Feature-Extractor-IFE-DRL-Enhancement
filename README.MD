# 🚀 **Unlocking Agent Understanding: Enhanced Interpretable Feature Extractor for Vision-Based Deep Reinforcement Learning**

Welcome to the official repository for our research project, **"Pay Attention to What and Where? Interpretable Feature Extractor for Vision-based Deep Reinforcement Learning"**, submitted to IJCNN 2025!

This project, a collaborative effort with [Ahsan Saleem](https://lnkd.in/ekiCtwaH), delves into making Deep Reinforcement Learning (DRL) agents more transparent and efficient, especially in vision-based tasks. We've significantly advanced the **Interpretable Feature Extractor (IFE)** to help us understand *what* and *where* a DRL agent is focusing its attention.

## ✨ **The Grand Vision: Why Interpretability Matters**

As DRL moves from research labs into real-world applications like autonomous systems, robotics, and edge devices, the "black box" nature of deep neural networks becomes a critical hurdle. For DRL agents to be trusted and widely adopted, we need to understand their decision-making process. This project directly addresses that challenge by providing tools to visualize and interpret an agent's internal workings.

Our core objective was to develop an IFE module that is not only highly accurate in revealing attention but also:
* **Scalable**: Easily adaptable to various DRL architectures and environments.
* **Lightweight**: Designed with efficiency in mind, minimizing computational overhead.
* **Efficient**: Capable of fast inference, suitable for real-time applications.

## 💡 **Our Innovative Approach**

We built upon existing concepts and introduced several key innovations to achieve our goals:

### 🖼️ **Visualizing Agent Attention: The IFE Module**
Our Interpretable Feature Extractor (IFE) is designed to provide clear, visual insights into the parts of the input a DRL agent prioritizes. By overlaying attention masks on the visual input, we can see in real-time what the agent is "looking at" to make its decisions. This is crucial for debugging, understanding biases, and building trust in autonomous systems.

### ⚙️ **Under the Hood: Key Architectural Enhancements**

* **From MLP to 1x1 Convolutional Attention:** We replaced traditional, heavy Multi-Layer Perceptron (MLP) attention mechanisms with a more streamlined **1x1 convolutional attention layer**. This significantly reduces the parameter count without compromising the preservation of spatial information, which is vital for vision-based tasks.
* **Depthwise Separable Convolutions:** To drastically cut down computational costs and improve efficiency, we integrated **depthwise separable convolutions** into our architecture. This innovative technique reduced the parameter count by up to **8 times** compared to standard convolutions!
* **Accelerated Inference:** Through these architectural optimizations, we achieved approximately **31% faster inference speeds**, reaching an impressive **46 Frames Per Second (FPS)**. This ensures that interpretability doesn't come at the cost of real-time performance.

## 📊 **Promising Results & ATARI Benchmark**

Our model was rigorously evaluated on the **ATARI benchmark**, a standard in Reinforcement Learning research. We confirmed that our attention masks are highly accurate when overlaid on visual input, offering clear and interpretable insights across all **57 ATARI games**.

### **Visual Proof in Action:**

Here are some glimpses of our interpretable attention masks:

| Boxing Result | Assault Result |
| :----------------------------------------------- | :------------------------------------------------- |
| ![Boxing Result](github_asset/boxing.gif?raw=true "Boxing Result") | ![Assault Result](github_asset/assault.gif?raw=true "Assault Result") |

This work specifically addresses the spatial interpretability problem in Explainable Deep Reinforcement Learning. Furthermore, our approach seamlessly integrates with existing DRL frameworks like [A3C LSTM](https://github.com/dgriff777/rl_a3c_pytorch), demonstrating its versatility and broad applicability. For an in-depth understanding of our interpretability evaluation and a live demo, please visit our dedicated project site: [Interpretable Feature Extractor Demo](https://sites.google.com/view/pay-attention-to-windows).

## 🚀 **Getting Started**

### **Prerequisites**

Our environment setup is tested on **Ubuntu 22.04**. Please be aware that you might encounter issues (e.g., unsupported packages) on other Linux distributions like CentOS Linux 7.9.2009 or Red Hat Enterprise Linux 7, and modifications may be necessary.

We primarily use `python=3.7`.

**1. Install System Dependencies:**

```bash
sudo apt install zlib1g-dev cmake unrar
```

**2. Install Python Packages:**

You can install all necessary Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, you can install them individually:

```bash
pip install wandb gym[atari]==0.18.0 imageio moviepy torchsummary tqdm rich procgen gym-retro torch stable_baselines3 atari_py==0.2.9 shimmy>=0.2.1
```

**3. Set up ATARI ROMs (if using `gym` Atari games):**

```bash
wget [http://www.atarimania.com/roms/Roms.rar](http://www.atarimania.com/roms/Roms.rar)
unrar x Roms.rar
python -m atari_py.import_roms .
```

**4. Set up `gym-retro` games (if desired):**

Follow the instructions provided [here](https://retro.readthedocs.io/en/latest/getting_started.html#importing-roms).

### **Training Your Own Model**

To train a model on a specific ATARI game (e.g., Boxing), run the following command:

```bash
python train.py --env_name gym:Boxing
```

This command will:
* Start the training process on the specified game.
* Use default hyperparameters (refer to `common/argp.py` for full details).
* Log all results to "Weights and Biases" for tracking.
* Save checkpoints to a directory named similarly to the Weights and Biases watermark. **Please check this directory for existing names to avoid overwriting previous runs.**

### **Pre-trained Models**

For your convenience, we provide pre-trained checkpoints for **Boxing, Assault, and SpaceInvaders** games. You can download them from this [Google Drive link](https://drive.google.com/drive/folders/1SMsP_MQN5sxevRQQ5ZmnklfT3o_C5tgQ?usp=drive_link). Once downloaded, place them in the `checkpoints` folder within this repository. If you require pre-trained models for other games, please feel free to contact the author.

### **Evaluating a Model**

To evaluate a trained model (or one of our pre-trained models) on a game like Boxing, use the command:

```bash
python eval.py --env_name gym:Boxing
```

This script will:
* Automatically load the final checkpoint (e.g., `checkpoint_49999680.pt`) from the `checkpoints` folder for the specified game.
* Initiate the evaluation process.
* Store the generated video of the evaluation in the `video` folder.

## 🙏 **Acknowledgements & Citations**

This project benefited immensely from existing open-source implementations and resources. We extend our gratitude to the authors of the following works:

* [https://github.com/schmidtdominik/Rainbow](https://github.com/schmidtdominik/Rainbow)
* OpenAI Baselines (particularly for preprocessing and Atari wrappers)
* [https://github.com/dgriff777/rl_a3c_pytorch](https://github.com/dgriff777/rl_a3c_pytorch)
* [https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py](https://github.com/vwxyzjn/cleanrl/blob/master/dqn_atari.py)
* [https://github.com/Kaixhin/Rainbow/](https://github.com/Kaixhin/Rainbow/)
* [https://github.com/Kaixhin/Rainbow/wiki/Matteo's-Notes](https://github.com/Kaixhin/Rainbow/wiki/Matteo's-Notes)

If you utilize this code or its concepts in your research or work, please cite our paper and ensure proper credit is given to the above-mentioned resources.

## 🤝 **Contributing**

While we do not actively expect pull requests for this specific research implementation, we highly value community input. If you discover any bugs, have suggestions for improvements, or encounter issues, please feel free to open an issue on this repository. Your feedback helps us refine and improve!
