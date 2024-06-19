#include <dgen.hh>

void Generator::save_shape_to_file(sf::Shape& shape, std::string fpath) {
    window_inst->draw(shape);
    window_inst->capture().saveToFile(fpath);
    window_inst->clear();
}

void Generator::save_label_to_file(uint8_t label, std::string fpath) {
    std::ofstream fout(fpath);
    fout << (int)label;
    fout.close();
}

void Generator::make_lnmap() { // label names map - сопоставляет лейблам строковые названия
    std::ofstream fout(this->path.main+"/setup.lnmap", std::ios::binary);
    auto byte_writer{[](auto value, std::ofstream& f) {f.write((char*)&value, sizeof(value));}};
    char sep = '\n';
    uint8_t label_sz = sizeof(uint8_t);
    byte_writer(label_sz, fout);
    byte_writer(sep, fout);
    for(auto& shape_class : shape_classes) {
        byte_writer(std::get<uint8_t>(shape_class), fout);
        fout << std::get<std::string>(shape_class);
        byte_writer(sep, fout);
    }
    fout.close();
}

void Generator::make_shape(size_t count, std::string name, bool is_train) {
    auto shape_ptr = std::find_if(shape_classes.begin(), shape_classes.end(), [name](std::tuple <std::string, size_t, uint8_t> elem){return (std::get<std::string>(elem) == name);});
        
    // suitable for shapes: triangle, pentagon, circle
    for (size_t i{0}; i < count; i++) {
        auto vertex = std::get<size_t>(*shape_ptr);
        auto label = std::get<uint8_t>(*shape_ptr);
        size_t rad = (std::rand()%(img_width/2-6))+5;
        sf::CircleShape polygon(rad, vertex); 
        polygon.move(std::rand()%(img_width-rad*2), std::rand()%(img_width-rad*2));
        save_shape_to_file(polygon, ((is_train)? path.train : path.test)+"/shape_"+std::to_string(i)+"_"+name+".png");
        save_label_to_file(label, ((is_train)? path.train : path.test)+"/label_"+std::to_string(i)+"_"+name+".label");
    }
}


void Generator::make_rect(size_t count, std::string name, bool is_train) {  // suitable for line, rect, rhombus
    auto shape_ptr = std::find_if(shape_classes.begin(), shape_classes.end(), [name](std::tuple <std::string, size_t, uint8_t> elem){return (std::get<std::string>(elem) == name);});

    for(size_t i{0};i < count;i++) {
        auto vertex = std::get<size_t>(*shape_ptr);
        auto lable = std::get<uint8_t>(*shape_ptr);
        if (std::get<std::string>(*shape_ptr) == "line") {
            int len = std::rand()%img_width + 4;
            sf::RectangleShape line(sf::Vector2f(len, 1));
            line.rotate(std::rand()%360);
            line.move(std::rand()%(img_width-len/2), std::rand()%(img_width-len/2));
            save_shape_to_file(line, ((is_train)? path.train : path.test)+"/shape_"+std::to_string(i)+"_"+name+".png");
            save_label_to_file(lable, ((is_train)? path.train : path.test)+"/label_"+std::to_string(i)+"_"+name+".label");
        } else if (std::get<std::string>(*shape_ptr) == "rhombus"){
            int len = std::rand()%(img_width-3) + 2;
            sf::RectangleShape rhombus(sf::Vector2f(len, len));
            rhombus.move(std::rand()%(img_height - len), std::rand()%(img_width - len));
            // rhombus.rotate(std::rand()%360);
            save_shape_to_file(rhombus, ((is_train)? path.train : path.test)+"/shape_"+std::to_string(i)+"_"+name+".png");
            save_label_to_file(lable, ((is_train)? path.train : path.test)+"/label_"+std::to_string(i)+"_"+name+".label");
        } else if (std::get<std::string>(*shape_ptr) == "rectangle") {
            int height = std::rand()%(img_height-3) + 2;
            int width = std::rand()%(img_width-3) + 2;
            if (height == width) {
                height += 1;
            }
            sf::RectangleShape rect(sf::Vector2f(height, width));
            rect.move(std::rand()%(img_height - height), std::rand()%(img_width - width));
            save_shape_to_file(rect, ((is_train)? path.train : path.test)+"/shape_"+std::to_string(i)+"_"+name+".png");
            save_label_to_file(lable, ((is_train)? path.train : path.test)+"/label_"+std::to_string(i)+"_"+name+".label");
        }
    }
}

void Generator::make_trapeze(size_t count, std::string name, bool is_train) {
    auto shape_ptr = std::find_if(shape_classes.begin(), shape_classes.end(), [name](std::tuple <std::string, size_t, uint8_t> elem){return (std::get<std::string>(elem) == name);});

    auto vertex = std::get<size_t>(*shape_ptr);
    auto lable = std::get<uint8_t>(*shape_ptr);       

    for (size_t i{0}; i < count; i++) {
        sf::ConvexShape trapeze{}; 
        int height = std::rand()%(img_height-3)+2;
        int a = std::rand()%(img_width-3)+2;
        int b = std::rand()%(img_width-3)+2;

        trapeze.setPointCount(4);

        int start = std::rand()%(img_width/2);
        int startb = std::rand()%(img_width-b)+b;
        trapeze.setPoint(0, sf::Vector2f(start, 0));
        trapeze.setPoint(1, sf::Vector2f(start + a, 0));
        trapeze.setPoint(2, sf::Vector2f(startb, height));
        trapeze.setPoint(3, sf::Vector2f(startb - b, height));

        save_shape_to_file(trapeze, ((is_train)? path.train : path.test)+"/shape_"+std::to_string(i)+"_"+name+".png");
        save_label_to_file(lable, ((is_train)? path.train : path.test)+"/label_"+std::to_string(i)+"_"+name+".label");
    }
}

void Generator::make_all_figures(size_t count, bool is_train) {
    std::vector<std::string> shapes{"triangle", "circle", "pentagon"};
    for (auto x: shapes) {
        make_shape(count, x, is_train);
    }
    shapes = {"line", "rectangle", "rhombus"};
    for (auto x: shapes) {
        make_rect(count, x, is_train);
    }
    make_trapeze(count, "trapeze", is_train);
}


// Пример использования 
int main() {
    std::srand(std::time(nullptr));
    Generator gen(27,27);
    gen.make_all_figures(10, false);    // for test
    gen.make_all_figures(10, true);     // for train
}


// g++ -o main.exe dataset.cc -lsfml-graphics -lsfml-window -lsfml-system