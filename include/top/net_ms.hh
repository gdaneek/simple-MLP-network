/**
 * \file net_ms.hh
 * Polymorphic model handler class <br>
 * requires the presence of methods for loading a model from a file
 * and saving it to a file. <br>
 * Uses network type restrictions.
 * Only classes inherited from the template polymorphic Net are allowed
 */

#ifndef NET_MODEL_SAVER
#define NET_MODEL_SAVER

#include <type_traits>

#include "top/net.hh"

/**
 *
 * Structure for checking whether the passed class is a
 * descendant of the base class with four template parameters
 */
template <template <typename, typename, typename, typename> class Base,
          typename Derived>
struct IsDerivedCheck {
 private:
  template <typename... Args>  // шаблон с переменным числом параметров
  static auto check(Base<Args...> *) -> std::true_type;  // пробуем создать Net
  static auto check(...) -> std::false_type;

 public:
  using type = decltype(check(std::declval<Derived *>()));
};

template <template <typename, typename, typename, typename> class Base,
          typename Derived>
concept DerivedFromNet = IsDerivedCheck<Base, Derived>::type::value;

/**
 * template class that implements a network model handler <br>
 */
template <typename NetType>
  requires DerivedFromNet<Net, NetType>
class ModelSaver {
 protected:
  const static inline uint64_t file_signature{0x10454c4946504c4d},
      model_end_lb{
          0x104f464e4954454e};  //< mlp file signature and model end mark values

 public:
  /**
   * Method for saving the network to an mlp file
   * \param net net to save
   * \param path file name to save
   * \param msg any message
   * \return path to the file where the network is saved
   */
  virtual std::string save_net_to_file(NetType &net,
                                       const std::string path = "",
                                       const std::string msg = "") = 0;
  /**
   * loads network from file
   * \param path mlp file with network
   * \return received network and message
   */
  virtual std::tuple<NetType, std::string> upload_net_from_file(
      std::string path) = 0;
};

#endif  // NET_MODEL_SAVER
