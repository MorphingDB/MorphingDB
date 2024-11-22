/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2024-08-13 13:37:22
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2024-08-20 22:52:54
 * @FilePath: /pgdl_basemodel_new/src/pgdl/model_selection.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "model_selection.h"
#include "spi_connection.h"
#pragma push_macro("Abs")
#pragma push_macro("snprintf")
#undef Abs
#undef snprintf
#include <onnxruntime_cxx_api.h> 
#pragma pop_macro("Abs")
#pragma pop_macro("snprintf")

extern "C" {
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/builtins.h"
#include "miscadmin.h"
}


const static std::vector<std::vector<float>> model_vector = {
    {5.08919555, 4.09464701, 2.06388991, 0.02279083, 3.24159773},
    {6.3125458,  2.0393437,  1.71464712, 1.69473217, 0.0       },
    {5.75990942, 2.36583906, 3.43353928, 1.61581757, 1.03587339},
    {5.88998286, 2.24559004, 1.85567387, 1.81298698, 1.21750197},
    {5.89371635, 2.131864,   1.95849534, 3.52834865, 0.47268591},
    {5.90091662, 2.2556092,  1.90265643, 2.05738155, 1.09460269},
    {5.90137434, 2.26012881, 1.96183597, 2.07524057, 1.40232097},
    {5.97324696, 2.21861088, 1.94552608, 2.12788017, 0.81659848},
    {3.36174447, 5.05727939, 1.67469931, 3.60929051, 5.36749255},
    {6.01349458, 1.65628063, 1.33960087, 1.86040519, 0.43786835},
    {1.38076849, 3.99390863, 5.65833173, 5.08092232, 8.48937111},
    {5.88631409, 1.76457918, 1.25113156, 1.57660384, 1.79521503},
    {5.31379553, 1.87179274, 1.52724113, 4.56104493, 1.26218823},
    {5.52093605, 1.93849537, 1.44889272, 2.36380101, 1.76276646},
    {5.47777617, 1.89950314, 1.37883369, 2.16068791, 0.55857653},
    {5.56101469, 1.89794416, 1.47708769, 2.3065968,  1.72578975},
    {3.58341096, 4.46840303, 1.58880889, 1.37850219, 6.22342154},
    {5.77642743, 0.61296288, 0.001691,   1.16870315, 0.85807453},
    {4.60776338, 0.87828427, 4.41321482, 1.8390199,  2.98943234},
    {4.76764257, 1.23920963, 0.08239659, 1.77823651, 3.84642254},
    {5.22417906, 0.63953153, 0.14856666, 3.81265227, 1.74084315},
    {4.8526845,  1.09788825, 0.0,        2.01636319, 3.43118694},
    {4.56458151, 0.81268128, 0.48028931, 2.03367418, 4.53717504},
    {4.96877227, 0.96185961, 0.53791155, 1.8546846,  3.04714918},
    {4.8075389,  4.25737762, 0.14685535, 0.87772465, 3.63464541},
    {5.99174986, 0.0002095,  0.83001742, 0.0,        0.97751868},
    {5.02499887, 0.60438441, 4.73644296, 0.67574263, 2.49298892},
    {5.31901198, 0.37067765, 0.75513206, 0.18109732, 3.78295299},
    {4.81854781, 0.38911067, 1.2722651,  4.6599276,  2.64072254},
    {4.955405,   0.57728781, 1.155203,   0.85726461, 3.77042166},
    {5.13596905, 0.47514542, 0.80579738, 0.43201026, 2.19264745},
    {5.10254978, 0.48149511, 1.14701644, 0.75880589, 3.40931684}
};

std::map<std::string, std::vector<float>> meta_scores_map = {
    {
        {
            "cifar10",
            {
                0.7661738704842023, 0.771644401870428, 0.7699229798319053, 0.7709685061007013,
                0.7707678495440637, 0.7718133758128598, 0.7719295453982815, 0.761474282710321,
                0.7284919611064494, 0.7298615963437036, 0.7294306107381583, 0.7296923750016491,
                0.7296421374157266, 0.7299039016792173, 0.7299329865973829, 0.7273153439624762,
                0.7125674791726065, 0.7153040797928514, 0.7144429487096469, 0.714965966974906,
                0.7148655897320785, 0.7153886079973376, 0.715446721137922, 0.7102165384853307,
                0.6734228537810598, 0.6843612466992479, 0.6809192427500883, 0.6830097850259582,
                0.6826085698417004, 0.6846991121175703, 0.6849313945926669, 0.6640259718339678
            }
        },
        {
            "oxford_pets",
            {
                0.8960053386213684, 0.9014758700075943, 0.8997544479690716, 0.9007999742378676,
                0.9005993176812299, 0.901644843950026, 0.9017610135354478, 0.8913057508474873,
                0.8583234292436156, 0.8596930644808698, 0.8592620788753246, 0.8595238431388152,
                0.8594736055528929, 0.8597353698163834, 0.8597644547345491, 0.8571468120996424,
                0.8423989473097728, 0.8451355479300175, 0.844274416846813, 0.8447974351120722,
                0.8446970578692448, 0.8452200761345039, 0.8452781892750882, 0.8400480066224969,
                0.8032543219182261, 0.8141927148364141, 0.8107507108872546, 0.8128412531631244,
                0.8124400379788667, 0.8145305802547365, 0.8147628627298331, 0.7938574399711341
            }
        },
        {
            "cub200",
            {
                0.9452092805180147, 0.9506798119042406, 0.9489583898657177, 0.9500039161345137,
                0.9498032595778761, 0.9508487858466722, 0.950964955432094, 0.9405096927441334,
                0.9075273711402618, 0.908897006377516, 0.9084660207719708, 0.9087277850354615,
                0.908677547449539, 0.9089393117130297, 0.9089683966311953, 0.9063507539962886,
                0.891602889206419, 0.8943394898266637, 0.8934783587434593, 0.8940013770087184,
                0.8939009997658909, 0.89442401803115, 0.8944821311717344, 0.8892519485191431,
                0.8524582638148723, 0.8633966567330603, 0.8599546527839008, 0.8620451950597706,
                0.8616439798755129, 0.8637345221513827, 0.8639668046264793, 0.8430613818677803
            }
        },
        {
            "caltech101",
            {
                0.9115248590075388, 0.9169953903937647, 0.9152739683552418, 0.9163194946240378,
                0.9161188380674002, 0.9171643643361963, 0.9172805339216181, 0.9068252712336575,
                0.8738429496297859, 0.8752125848670401, 0.8747815992614949, 0.8750433635249856,
                0.8749931259390631, 0.8752548902025538, 0.8752839751207194, 0.8726663324858127,
                0.857918467695943, 0.8606550683161878, 0.8597939372329834, 0.8603169554982425,
                0.860216578255415, 0.8607395965206741, 0.8607977096612585, 0.8555675270086672,
                0.8187738423043964, 0.8297122352225844, 0.8262702312734249, 0.8283607735492947,
                0.827959558365037, 0.8300501006409068, 0.8302823831160034, 0.8093769603573044
            }
        },
        {
            "stanford_dogs",
            {
                0.902453054639166, 0.9079235860253918, 0.9062021639868689, 0.907247690255665,
                0.9070470336990274, 0.9080925599678235, 0.9082087295532453, 0.8977534668652847,
                0.8647711452614131, 0.8661407804986673, 0.865709794893122, 0.8659715591566127,
                0.8659213215706902, 0.866183085834181, 0.8662121707523466, 0.86359452811744,
                0.8488466633275703, 0.851583263947815, 0.8507221328646106, 0.8512451511298698,
                0.8511447738870421, 0.8516677921523013, 0.8517259052928856, 0.8464957226402944,
                0.8097020379360236, 0.8206404308542116, 0.8171984269050521, 0.819288969180922,
                0.8188877539966641, 0.820978296272534, 0.8212105787476307, 0.8003051559889316
            }
        },
        {
            "nabird",
            {
                1.018773713088766, 1.0242442444749917, 1.0225228224364689, 1.023568348705265,
                1.0233676921486274, 1.0244132184174235, 1.0245293880028452, 1.0140741253148846,
                0.9810918037110131, 0.9824614389482673, 0.982030453342722, 0.9822922176062127,
                0.9822419800202903, 0.982503744283781, 0.9825328292019465, 0.9799151865670399,
                0.9651673217771702, 0.967903922397415, 0.9670427913142106, 0.9675658095794697,
                0.9674654323366422, 0.9679884506019013, 0.9680465637424857, 0.9628163810898943,
                0.9260226963856235, 0.9369610893038116, 0.933519085354652, 0.9356096276305219,
                0.9352084124462641, 0.937298954722134, 0.9375312371972306, 0.9166258144385315
            }
        },
        {
            "mean",
            {
                0.906690019393176, 0.9121605507794018, 0.9104391287408791, 0.9114846550096751,
                0.9112839984530374, 0.9123295247218335, 0.9124456943072553, 0.9019904316192948,
                0.8690081100154231, 0.8703777452526773, 0.8699467596471321, 0.8702085239106228,
                0.8701582863247004, 0.8704200505881909, 0.8704491355063566, 0.8678314928714499,
                0.8530836280815803, 0.8558202287018251, 0.8549590976186207, 0.8554821158838798,
                0.8553817386410523, 0.8559047569063114, 0.8559628700468958, 0.8507326873943044,
                0.8139390026900336, 0.8248773956082216, 0.8214353916590621, 0.823525933934932,
                0.8231247187506742, 0.8252152610265441, 0.8254475435016407, 0.8045421207429416
            }
        }
    }
};

const static std::vector<std::string> all_architectures = {
    "resnet50",
    "resnet18",
    "googlenet",
    "alexnet",
};

const static std::vector<std::string> source_datasets = {
    "nabird",
    "oxford_pets",
    "cub200",
    "caltech101",
    "stanford_dogs",
    "voc2007",
    "cifar10",
    "imagenet"
};

const static std::vector<std::string> models_name = []() {
    std::vector<std::string> models_name;
    for (const auto& arch : all_architectures) {
        for (const auto& dataset : source_datasets) {
            std::string model_name = arch + "%" + dataset;
            models_name.push_back(model_name);
        }
    }
    return models_name;
}();

std::vector<float> TensorToVector(const torch::Tensor& tensor) {
    return std::vector<float>(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
}

torch::Tensor MatToTensor(const cv::Mat& image) {
    if (image.empty()) {
        //std::cerr << "Empty image provided!" << std::endl;
        return torch::Tensor();
    }

    if (image.channels() != 3) {
        //std::cerr << "Image does not have 3 channels!" << std::endl;
        return torch::Tensor();
    }

    int height = image.rows;
    int width = image.cols;
    int channels = image.channels();

    torch::Tensor tensor_image = torch::zeros({channels, height, width}, torch::kFloat32);

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                tensor_image[c][h][w] = static_cast<float>(image.at<cv::Vec3b>(h, w)[c]) / 255.0;
            }
        }
    }

    return tensor_image;
}


torch::Tensor ModelSelection::GetForwardClip(const std::vector<std::string>& data_list, std::string visual_model_path) {
    torch::Device device(torch::kCPU);
    torch::jit::script::Module model = torch::jit::load(visual_model_path);
    model.to(device);
    model.eval();

    std::vector<torch::Tensor> all_feats;
    torch::NoGradGuard no_grad;
    for (const auto& image_path : data_list) {
        torch::Tensor image = Preprocess(image_path, default_n_px).to(device);
        torch::Tensor feats = model.forward({image}).toTensor().to(device);
        all_feats.push_back(feats);
    }

    torch::Tensor result = torch::cat(all_feats, 0);
    return result;
}


torch::Tensor ModelSelection::Preprocess(const std::string& image_path, int n_px) {
    // Read image using OpenCV
    cv::Mat image = cv::imread(image_path);

    if (image.empty()) {
        //std::cerr << "Failed to read image: " << image_path << std::endl;
        //exit(-1);
    }

    // Resize image (the shorter edge scaled to n_px, bicubic interpolation)
    int new_rows = 0, new_cols = 0;
    if (image.rows > image.cols) {
        new_cols = n_px;
        new_rows = n_px * image.rows / image.cols;
    } else {
        new_rows = n_px;
        new_cols = n_px * image.cols / image.rows;
    }
    cv::resize(image, image, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_CUBIC);

    // Center crop
    int crop_x = (image.cols - n_px) / 2;
    int crop_y = (image.rows - n_px) / 2;
    cv::Rect roi(crop_x, crop_y, n_px, n_px);

    image = image(roi);

    // Convert BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Convert Mat to Tensor and permute dimensions to match [1, 3, n_px, n_px]
    // torch::Tensor tensor_image = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte);
    // tensor_image = tensor_image.permute({2, 0, 1});
    torch::Tensor tensor_image = MatToTensor(image); // use our own function (from_blob doesn't work, why?)

    // Normalize (the parameters are fixed)
    tensor_image = torch::data::transforms::Normalize<>(
        {0.48145466, 0.4578275, 0.40821073},
        {0.26862954, 0.26130258, 0.27577711}
    )(tensor_image);

    // Add batch dimension
    tensor_image = tensor_image.unsqueeze(0);  // Add batch dimension

    return tensor_image;
}

std::vector<std::string> ModelSelection::GetDataList(const std::string& table_name,
                                         const std::string& col_name,
                                         const int& sample_size)
{
    std::vector<std::string> data_list;

    // get image path list
    SPIConnector spi_connector;
    std::string sql_str = "SELECT " + col_name + " FROM " + table_name + " LIMIT " + std::to_string(sample_size);
    SPISqlWrapper sql(spi_connector, sql_str, 0);
    if(sql.Execute()){
        if(SPI_processed != 10){
            
        }
    }
    // get result
    int total_tuples = sample_size;
    for(int i=0; i<total_tuples; i++){
        HeapTuple tuple = SPI_tuptable->vals[i];
        std::string image_path = SPI_getvalue(tuple, SPI_tuptable->tupdesc, 1);
        data_list.push_back(image_path);
    }
    return data_list;
}
    



std::string ModelSelection::SelectModel(const std::string& table_name,
                                         const std::string& col_name,
                                         const int& sample_size,
                                         std::string dataset)
{
    if(dataset.empty()){
        dataset = "mean";
    }


    if (meta_scores_map.find(dataset) == meta_scores_map.end()) {
        //std::cerr << "Dataset not found in meta_scores_map" << std::endl;
        //exit(-1);
    }
    std::vector<float> meta_scores = meta_scores_map[dataset];
    
    int input_size = sample_size;
    std::vector<std::string> data_list = GetDataList(table_name, 
                                                     col_name, 
                                                     sample_size);

    // get forward from clip
    torch::Tensor predict_feats = GetForwardClip(data_list, visual_model_path);

    // load regression model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
    Ort::SessionOptions session_options;
    Ort::Session session(env, regression_model_path.c_str(), session_options);
    Ort::AllocatedStringPtr input_node_name_ptr = session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    const char* input_node_name = input_node_name_ptr.get();
    std::vector<int64_t> input_shape = {input_size, 768};
    std::vector<float> input_tensor_values = TensorToVector(predict_feats);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    Ort::AllocatedStringPtr output_node_name_ptr = session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    const char* output_node_name = output_node_name_ptr.get();
    std::vector<const char*> input_names = {input_node_name};
    std::vector<const char*> output_names = {output_node_name};

    // predict and calculate mean of the output tensor along axis 0
    std::vector<int64_t> output_shape = {input_size, regression_output_dim};
    auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
    float* float_array = output_tensor[0].GetTensorMutableData<float>();
    torch::Tensor scores = torch::from_blob(float_array, torch::IntArrayRef(output_shape));

    torch::Tensor score = torch::mean(scores, 0);

    // calculate trans_scores
    torch::Tensor score_transposed = torch::unsqueeze(score, /*dim=*/1);
    std::vector<torch::Tensor> rows;
    for (const auto& row : model_vector) {
        rows.push_back(torch::tensor(row));
    }
    torch::Tensor model_vector_tensor = torch::stack(rows);
    torch::Tensor trans_scores_tensor = torch::mm(model_vector_tensor, score_transposed);
    std::vector<float> trans_scores = TensorToVector(trans_scores_tensor);

    // calculate final scores and find the max index
    double alpha = 0.5;
    std::vector<float> final(trans_scores.size());
    for (size_t i = 0; i < trans_scores.size(); ++i) {
        //final[i] = (1 - alpha) * trans_scores[i] + alpha * meta_scores[i];
        final[i] = trans_scores[i];
    }
    auto max_index = std::distance(final.begin(), std::max_element(final.begin(), final.end()));

    // return the selected model name
    return models_name[max_index];
}
