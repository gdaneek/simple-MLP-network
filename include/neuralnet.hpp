#ifndef BASE_HPP
#define BASE_HPP
#include <vector>
#include <cstdlib>
#include <set>
                                                                                // WARNING: плохие модификаторы доступа, их позже сделаю правильно
class NeuralLink {                                                              // класс весов. 
                                                                                // --
public:                                                                         // --
   std::vector<std::vector<double>> weight;                                     // Матрица m x n связывающая слои длины m и n 
   NeuralLink(size_t input_layer_size, size_t output_layer_size);               // Конструктор. Обязательно указываются размеры слоёв 
   std::vector<double>& operator [](int64_t i);                                 // Перегрузка на []. Выкидывает ссылку на n весов нейрона 0 <= i <= m 
   std::vector<std::vector<double>>::iterator begin();                          // Итератор на начало матрицы весов. Для обхода класса
   std::vector<std::vector<double>>::iterator end();                            // Итератор на конец матрицы весов
};                                                                              // --
class Neuron {                                                                  // Класс нейрона. shift не добавил пока что никуда, с ним отдельная работа
public:                                                                         // --
    double value;                                                               // Значение нейрона
    double shift;                                                               // Сдвиг значения
    Neuron(double value = 0);                                                   // Конструктор. По умолчанию нейрон собирается с нулём
   ~Neuron();                                                                   // Деструктор
    void operator =(double value);                                              // Перегрузка на присваивание.
    void operator +=(double value);                                             // Перегрузка на суммирование.
    double get_value();                                                         // << Удалить >>
};                                                                              // --
class Layer {                                                                   // Класс слоя
    std::vector<Neuron> neurons;                                                // Слой это вектор нейронов
public:                                                                         // --
    Layer(size_t size);                                                         // Конструктор. Параметр - размер слоя (число нейронов)
    ~Layer();                                                                   // Деструктор
    std::vector<double> operator *(std::vector<double>& weight);                // Перегрузка на умножение. Умножает на вектор весов
    std::vector<double> operator *(NeuralLink& weight);                         // Перегрузка на умножение. Умножает на матрицу нейролинков и возвращает вектор 
    Layer& operator = (std::vector<double> values);                             // Слою можно присваивать вектор
    size_t size();                                                              // Размер слоя - число нейронов
    double operator [](int64_t i);                                              // Возвращает нейрон по индексу в слое
    std::vector<Neuron>::iterator begin();                                      // Для обхода слоя по нейронам
    std::vector<Neuron>::iterator end();                                        // Для обхода слоя по нейронам
};                                                                              // --
                                                                                // --
namespace activations {                                                         // Пространство функций активации
    struct ActivationFunc {                                                     // Абстрактный класс для наследования функцией-классом активации
        virtual void operator ()(Neuron* neuron, ... ) = 0;                     // Описывает применение функции к нейрону
        virtual void operator ()(Layer* layer, ... ) = 0;                       // Описывает применение функции к слою
    };                                                                          // --
    struct Empty : public ActivationFunc {                                      // Реализация пустой функции, не меняющей ни слоя, ни нейрона
      void operator ()(Neuron* neuron, ... ) override;                          // --
      void operator ()(Layer* layer, ... ) override;                            // --
    };                                                                          // --
    struct ReLU: public ActivationFunc {                                        // Реализация relu. Применяется для скрытых слоёв
      void operator ()(Neuron* neuron, ... ) override;                          // --
      void operator ()(Layer* layer, ... ) override;                            // --
    };                                                                          // --
    struct Tanh: public ActivationFunc {                                        // Реализация гиперболического тангенса. Для скрытых слоёв
      void operator ()(Neuron* neuron, ... ) override;                          // --
      void operator ()(Layer* layer, ... ) override;                            // --
    };                                                                          // --
    struct Sigmoid: public ActivationFunc {                                     // Реализация сигмоиды. Для скрытых слоёв
      void operator ()(Neuron* neuron, ... ) override;                          // --
      void operator ()(Layer* layer, ... ) override;                            // --
    };                                                                          // --
    struct Softmax : public ActivationFunc {                                    // Реализация софтмакс. Для выходного слоя
      double exp_sum{0};                                                        // --
      void operator ()(Neuron* neuron, ... ) override;                          // --
      void operator ()(Layer* layer, ... ) override;                            // --
    };                                                                          // --
                                                                                // --
    double derivative(double x0, ActivationFunc* f, double accuracy);           // Первая производная для функции активации в точке x0
                                                                                // --
    double derivative(Neuron &neuron, ActivationFunc* f, double accuracy);      // Первая производная для нейрона (по его значению) в точке х0
};                                                                              // --
extern activations::Empty empty;                                                // --
extern activations::ReLU relu;                                                  // --
extern activations::Tanh tanh_;                                                 // --
extern activations::Sigmoid sigmoid;                                            // -- 
extern activations::Softmax softmax;                                            // --
class Net {                                                                     // Класс нейронной сети
    size_t layer_indexer;                                                       // индексатор слоев. Нужен для позиционирования слоёв относительно друг друга
    std::set<std::pair<size_t,std::pair<size_t, activations::ActivationFunc*>>> layer_table; // Таблица слоёв, задаваемая пользователем. По ней собирается сеть
    std::vector<Layer> layers;                                                  // Вектор слоёв нейронов
public:                                                                         // --
    std::vector<NeuralLink> NeuralLinks;                                        // Все нейронные связи
    std::vector<activations::ActivationFunc*> activations;                      // Вектор функций активации
    void add_layer(size_t neuron_count, activations::ActivationFunc& activation, size_t layer_index); // добавляет слой в таблицу
    void add_layer(size_t neuron_count, activations::ActivationFunc& activation); 
    Net(size_t input_size, size_t output_size, activations::ActivationFunc& activation_in, activations::ActivationFunc& activation_out);
    void make();                                                                // Создать сеть по  таблице
    size_t size();                                                              // Размер сети - число слоёв
    Layer& operator [](int64_t i);                                              // Получить слой по индексу
    void set_input(std::vector<double>& values);                                // Устанавливает значения на входной слой
    Layer& calc_output();                                                       // Рассчитывает выход
    Layer& calc_output(std::vector<double>& input_values);                      //
    std::pair<size_t, double> result();                                         // Индекс и значение нейрона, содержащего наиболее большое значение
    std::vector<Layer>::iterator begin();                                       // Итерация по сети - обход по слоям
    std::vector<Layer>::iterator end();
   
    
    

};

#endif
