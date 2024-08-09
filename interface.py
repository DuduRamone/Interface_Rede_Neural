import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from tkinter import Tk, filedialog

# Classe DataHandler
class DataHandler:
    def __init__(self):
        self.file_name = None
        self.data = None
        self.x = None
        self.y = None
        self.scaler = None
        self.label_encoders = {}

    def load_data(self, file_name):
        self.file_name = file_name
        column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
                        "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        self.data = pd.read_csv(self.file_name, header=0, sep=',', na_values="?", skipinitialspace=True)
        self.data.dropna(inplace=True)

    def preprocess_data(self, data_drop="income"):
        for column in self.data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            self.label_encoders[column] = le
        self.x = self.data.drop(data_drop, axis=1)
        self.y = self.data[data_drop]

        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(self.x)

    def get_train_test_data(self, test_size=0.2):
        return train_test_split(self.x, self.y, test_size=test_size, random_state=42)

# Classe NeuralNetwork
class NeuralNetwork:
    def __init__(self):
        self.model = None

    def configure_model(self, input_shape, layers_config, optimizer='adam'):
        self.model = Sequential()
        self.model.add(Dense(units=layers_config[0]['units'], activation=layers_config[0]['activation'], input_shape=(input_shape,)))
        for layer in layers_config[1:]:
            self.model.add(Dense(units=layer['units'], activation=layer['activation']))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    def train_model(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=32, callback=None):
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[callback], verbose=0)

class TrainingCallback(Callback):
    def __init__(self, update_terminal):
        super().__init__()
        self.update_terminal = update_terminal

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.update_terminal(f"Epoch {epoch+1}: accuracy={logs.get('accuracy'):.4f}, val_accuracy={logs.get('val_accuracy'):.4f}, loss={logs.get('loss'):.4f}, val_loss={logs.get('val_loss'):.4f}")

def plot_to_surface(fig, width, height):
    fig.set_size_inches(width / fig.dpi, height / fig.dpi)
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    return pygame.image.fromstring(raw_data, size, "RGB")

def create_plot(history, metric, width, height):
    fig, ax = plt.subplots()
    ax.plot(history.history[metric], label=f'Train {metric}')
    ax.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    return plot_to_surface(fig, width, height)

def create_confusion_matrix(y_true, y_pred, width, height):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    return plot_to_surface(fig, width, height)

# Funções auxiliares para Pygame
def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    text_rect.topleft = (x, y)
    surface.blit(text_obj, text_rect)

def draw_input_box(x, y, w, h, text, active):
    input_box = pygame.Rect(x, y, w, h)
    color = pygame.Color('lightskyblue3') if active else pygame.Color('gray')
    txt_surface = font.render(text, True, color)
    width = max(w, txt_surface.get_width() + 10)
    input_box.w = width
    screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
    pygame.draw.rect(screen, color, input_box, 2)
    return input_box

def draw_slider(x, y, w, h, value):
    pygame.draw.rect(screen, pygame.Color('grey'), (x, y, w, h), 2)
    pygame.draw.rect(screen, pygame.Color('lightskyblue3'), (x, y, value, h))
    return pygame.Rect(x, y, w, h)

def draw_button(text, font, color, surface, x, y, w, h):
    button_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, color, button_rect)
    txt_surface = font.render(text, True, (255, 255, 255))
    text_rect = txt_surface.get_rect(center=(x + w / 2, y + h / 2))
    surface.blit(txt_surface, text_rect)
    return button_rect

def draw_combobox(surface, font, x, y, w, h, options, selected_index):
    combobox_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, pygame.Color('gray'), combobox_rect, 2)
    if selected_index >= 0:
        text = options[selected_index]
    else:
        text = "Selecionar"
    txt_surface = font.render(text, True, pygame.Color('lightskyblue3'))
    surface.blit(txt_surface, (combobox_rect.x + 5, combobox_rect.y + 5))
    return combobox_rect

def draw_terminal(output, font, surface, x, y, w, h):
    terminal_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, pygame.Color('black'), terminal_rect)
    pygame.draw.rect(surface, pygame.Color('white'), terminal_rect, 2)
    
    y_offset = y + 10
    for line in output[-(h // 20):]:
        txt_surface = font.render(line, True, pygame.Color('white'))
        if txt_surface.get_width() > w - 20:
            txt_surface = pygame.transform.scale(txt_surface, (w - 20, txt_surface.get_height()))
        surface.blit(txt_surface, (x + 10, y_offset))
        y_offset += 20

def open_file_dialog():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path

# Inicializando Pygame
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption('Configuração da Rede Neural')
font = pygame.font.Font(None, 28)  # Reduzindo o tamanho da fonte
clock = pygame.time.Clock()

# Variáveis de configuração
epochs = 50
batch_size = 32
train_test_split_ratio = 70
hidden_layers = 3
activation_functions = ['relu', 'sigmoid', 'tanh', 'linear']
optimization_functions = ['adam', 'sgd', 'rmsprop']
selected_optimization = 0
layers_config = [{'units': 64, 'activation': 'relu'}, {'units': 32, 'activation': 'relu'}, {'units': 16, 'activation': 'relu'}]
training_output = []

input_text_epochs = str(epochs)
input_text_batch = str(batch_size)
input_text_layers_count = str(hidden_layers)
input_active_epochs = False
input_active_batch = False
input_active_layers_count = False
input_active_layers = [False] * hidden_layers
input_text_layers = [str(layer['units']) for layer in layers_config]
selected_activation = [0] * hidden_layers

scroll_offset = 0
accuracy_plot = None
loss_plot = None
confusion_matrix_plot = None
data_handler = DataHandler()

def update_terminal(message):
    training_output.append(message)
    draw_interface()

# Função para desenhar a interface
def draw_interface(history=None):
    global accuracy_plot, loss_plot, confusion_matrix_plot
    screen.fill((30, 30, 30))
    draw_text('Configuração da Rede Neural', font, (255, 255, 255), screen, 20, 20 - scroll_offset)
    
    load_data_button = draw_button('Carregar Dataset', font, pygame.Color('blue'), screen, 20, 60 - scroll_offset, 200, 50)
    
    draw_text('Número de camadas ocultas:', font, (255, 255, 255), screen, 20, 130 - scroll_offset)
    layers_count_input_box = draw_input_box(300, 130 - scroll_offset, 50, 32, input_text_layers_count, input_active_layers_count)
    
    layer_input_boxes = []
    activation_comboboxes = []
    for i in range(hidden_layers):
        y_offset = 170 + i * 50 - scroll_offset
        draw_text(f'Camada {i + 1} - Neurônios:', font, (255, 255, 255), screen, 20, y_offset)
        input_box = draw_input_box(300, y_offset, 140, 32, input_text_layers[i], input_active_layers[i])
        combobox_rect = draw_combobox(screen, font, 500, y_offset, 140, 32, activation_functions, selected_activation[i])
        layer_input_boxes.append(input_box)
        activation_comboboxes.append(combobox_rect)
    
    draw_text('Número de épocas:', font, (255, 255, 255), screen, 20, 170 + hidden_layers * 50 - scroll_offset)
    epochs_input_box = draw_input_box(300, 170 + hidden_layers * 50 - scroll_offset, 140, 32, input_text_epochs, input_active_epochs)
    
    draw_text('Tamanho do batch:', font, (255, 255, 255), screen, 20, 230 + hidden_layers * 50 - scroll_offset)
    batch_size_input_box = draw_input_box(300, 230 + hidden_layers * 50 - scroll_offset, 140, 32, input_text_batch, input_active_batch)
    
    draw_text(f'Divisão treino/teste: {train_test_split_ratio}%', font, (255, 255, 255), screen, 20, 290 + hidden_layers * 50 - scroll_offset)
    train_test_slider_rect = draw_slider(20, 330 + hidden_layers * 50 - scroll_offset, 200, 20, train_test_split_ratio * 2)
    pygame.draw.circle(screen, pygame.Color('lightskyblue3'), (20 + train_test_split_ratio * 2, 340 + hidden_layers * 50 - scroll_offset), 10)
    
    draw_text('Otimização:', font, (255, 255, 255), screen, 20, 390 + hidden_layers * 50 - scroll_offset)
    optimization_combobox = draw_combobox(screen, font, 300, 390 + hidden_layers * 50 - scroll_offset, 140, 32, optimization_functions, selected_optimization)

    train_button_rect = draw_button('Treinar', font, pygame.Color('green'), screen, 20, 450 + hidden_layers * 50 - scroll_offset, 120, 50)
    
    draw_terminal(training_output, font, screen, 20, 520 + hidden_layers * 50 - scroll_offset, 800, 180)  # Aumentando a largura da caixa do terminal

    if accuracy_plot and loss_plot and confusion_matrix_plot:
        screen.blit(accuracy_plot, (850, 20))
        screen.blit(loss_plot, (850, 250))
        screen.blit(confusion_matrix_plot, (850, 480))

    pygame.display.flip()
    return load_data_button, layers_count_input_box, train_button_rect, layer_input_boxes, activation_comboboxes, train_test_slider_rect, optimization_combobox, epochs_input_box, batch_size_input_box

# Loop principal
running = True
while running:
    load_data_button, layers_count_input_box, train_button_rect, layer_input_boxes, activation_comboboxes, train_test_slider_rect, optimization_combobox, epochs_input_box, batch_size_input_box = draw_interface()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if load_data_button.collidepoint(event.pos):
                file_path = open_file_dialog()
                if file_path:
                    data_handler.load_data(file_path)
                    update_terminal(f"Dataset carregado: {file_path}")
            if layers_count_input_box.collidepoint(event.pos):
                input_active_layers_count = True
                input_active_epochs = False
                input_active_batch = False
            if train_test_slider_rect.collidepoint(event.pos):
                train_test_split_ratio = int((event.pos[0] - 20) / 2)
            if optimization_combobox.collidepoint(event.pos):
                selected_optimization = (selected_optimization + 1) % len(optimization_functions)
            if train_button_rect.collidepoint(event.pos):
                # Atualizar valores de epochs e batch_size com os inputs
                epochs = int(input_text_epochs)
                batch_size = int(input_text_batch)
                
                # Atualizar a configuração das camadas ocultas
                for i in range(hidden_layers):
                    layers_config[i]['units'] = int(input_text_layers[i])
                    layers_config[i]['activation'] = activation_functions[selected_activation[i]]
                
                # Imprimir as variáveis no terminal
                training_output.append(f"Épocas: {epochs}")
                training_output.append(f"Tamanho do Batch: {batch_size}")
                training_output.append(f"Divisão Treino/Teste: {train_test_split_ratio}%")
                training_output.append(f"Otimização: {optimization_functions[selected_optimization]}")
                training_output.append("Configuração das Camadas:")
                for i, layer in enumerate(layers_config):
                    training_output.append(f"  Camada {i + 1}: {layer['units']} neurônios, ativação {layer['activation']}")
                
                # Carregar e preprocessar os dados
                data_handler.preprocess_data(data_drop="income")
                x_train, x_test, y_train, y_test = data_handler.get_train_test_data(test_size=(100 - train_test_split_ratio) / 100)
                
                # Configurar e treinar a rede neural
                nn = NeuralNetwork()
                nn.configure_model(input_shape=x_train.shape[1], layers_config=layers_config, optimizer=optimization_functions[selected_optimization])
                training_callback = TrainingCallback(update_terminal)
                history = nn.train_model(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size, callback=training_callback)

                # Criar e exibir os gráficos de acurácia, perda e matriz de confusão
                accuracy_plot = create_plot(history, 'accuracy', 600, 200)
                loss_plot = create_plot(history, 'loss', 600, 200)
                y_pred = (nn.model.predict(x_test) > 0.5).astype("int32")
                confusion_matrix_plot = create_confusion_matrix(y_test, y_pred, 600, 200)
                
            for i, box in enumerate(layer_input_boxes):
                if box.collidepoint(event.pos):
                    input_active_layers = [False] * hidden_layers
                    input_active_layers[i] = True
                else:
                    input_active_layers[i] = False
            for i, combobox in enumerate(activation_comboboxes):
                if combobox.collidepoint(event.pos):
                    selected_activation[i] = (selected_activation[i] + 1) % len(activation_functions)
            if epochs_input_box.collidepoint(event.pos):
                input_active_epochs = True
                input_active_batch = False
            elif batch_size_input_box.collidepoint(event.pos):
                input_active_batch = True
                input_active_epochs = False
            else:
                input_active_epochs = False
                input_active_batch = False
        elif event.type == pygame.KEYDOWN:
            if input_active_epochs:
                if event.key == pygame.K_RETURN:
                    input_active_epochs = False
                elif event.key == pygame.K_BACKSPACE:
                    input_text_epochs = input_text_epochs[:-1]
                else:
                    input_text_epochs += event.unicode
            if input_active_batch:
                if event.key == pygame.K_RETURN:
                    input_active_batch = False
                elif event.key == pygame.K_BACKSPACE:
                    input_text_batch = input_text_batch[:-1]
                else:
                    input_text_batch += event.unicode
            if input_active_layers_count:
                if event.key == pygame.K_RETURN:
                    input_active_layers_count = False
                    hidden_layers = max(1, int(input_text_layers_count))
                    layers_config = [{'units': 10, 'activation': 'relu'}] * hidden_layers
                    input_text_layers = ['10'] * hidden_layers
                    selected_activation = [0] * hidden_layers
                    input_active_layers = [False] * hidden_layers
                elif event.key == pygame.K_BACKSPACE:
                    input_text_layers_count = input_text_layers_count[:-1]
                else:
                    input_text_layers_count += event.unicode
            for i in range(hidden_layers):
                if input_active_layers[i]:
                    if event.key == pygame.K_RETURN:
                        input_active_layers[i] = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text_layers[i] = input_text_layers[i][:-1]
                    else:
                        input_text_layers[i] += event.unicode
        elif event.type == pygame.MOUSEWHEEL:
            scroll_offset = max(0, scroll_offset - event.y * 20)

pygame.quit()
