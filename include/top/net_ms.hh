#ifndef NET_MODEL_SAVER
#define NET_MODEL_SAVER

#include "net.hh"
#include "activations.hh"
#include <concepts>
#include <type_traits>

template<template<typename, typename, typename, typename> class BaseTemplate, typename T>
struct is_derived_from_template_impl {
    
private:
    template<typename... Args>
    static auto test(const BaseTemplate<Args...>*) -> std::true_type;

    static auto test(...) -> std::false_type;  
public:
  using type = decltype(test(std::declval<T*>()));
};

template<template<typename, typename, typename, typename> class BaseTemplate, typename T>
using is_derived_from_template = typename is_derived_from_template_impl<BaseTemplate, T>::type;

// Концепт, который проверяет, что T является производным от шаблонного класса BaseTemplate
template<typename T, template<typename, typename, typename, typename> class BaseTemplate>
concept DerivedFromTemplate4 = is_derived_from_template<BaseTemplate, T>::value;


/**
 * template class that implements a network model handler <br>
*/
template<typename net_t>
requires DerivedFromTemplate4<net_t, Net>
class ModelSaver {
  protected:
    const static inline uint64_t file_signature{0x10454c4946504c4d},
                                 net_info_label{0x104f464e4954454e};
  public:
    virtual std::string save_net_to_file(net_t &net, const std::string path = "", const std::string msg = "") = 0;
    virtual std::tuple<net_t, std::string> upload_net_from_file(std::string path) = 0;
};

#endif
