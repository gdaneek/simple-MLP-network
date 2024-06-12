#ifndef NET_MODEL_SAVER
#define NET_MODEL_SAVER

#include "net.hh"
#include <type_traits>

template<template<typename, typename, typename, typename> class Base, typename Derived>
struct is_derived_from_template_impl {
private:
  template<typename... Args>                                        // шаблон с переменным числом параметров
  static auto check(Base<Args...>*) -> std::true_type;               // пробуем создать Net
  static auto check(...) -> std::false_type;  
public:
  using type = decltype(check(std::declval<Derived*>()));
};

template< template<typename, typename, typename, typename> class Base, typename Derived>
concept DerivedFromNet = is_derived_from_template_impl<Base, Derived>::type::value;


/**
 * template class that implements a network model handler <br>
*/
template<typename NetType> requires DerivedFromNet<Net, NetType>
class ModelSaver {
  protected:
    const static inline uint64_t file_signature{0x10454c4946504c4d},
                                 net_info_label{0x104f464e4954454e};
  public:
    virtual std::string save_net_to_file(NetType &net, const std::string path = "", const std::string msg = "") = 0;
    virtual std::tuple<NetType, std::string> upload_net_from_file(std::string path) = 0;
};

#endif
