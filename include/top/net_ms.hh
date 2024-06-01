#ifndef NET_MODEL_SAVER
#define NET_MODEL_SAVER

#include "net.hh"
#include "activations.hh"

/**
 * template class that implements a network model handler <br>
*/
template<class NetType>
class ModelSaver {
  protected:
    const static inline uint64_t file_signature{0x10454c4946504c4d},
                                 net_info_label{0x104f464e4954454e};
  public:
    virtual std::string save_net_to_file(NetType &net, const std::string path = "", const std::string msg = "") = 0;
    virtual NetType netmaker(std::vector<std::tuple<size_t, activations::fptr>>& layers, weight_vector& weights, std::vector<neuron_t>& shifts) = 0;
    virtual std::tuple<NetType, std::string> upload_net_from_file(std::string path) = 0;
};

#endif