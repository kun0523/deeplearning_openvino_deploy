// copyfiles.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <filesystem>
#include <regex>

void copy_files(const std::string& src_file, const std::string& dst_dir, const std::string& save_name)
{
	try {
		std::filesystem::path src{ src_file };
		std::filesystem::path dst{ dst_dir + "/" + save_name };

		if (!std::filesystem::exists(src))
			return;
		if (!std::filesystem::exists(dst_dir))
			std::filesystem::create_directories(dst_dir);

		std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
	}
	catch (const std::filesystem::filesystem_error& e)
	{
		std::cerr << e.what() << std::endl;
	}
}

int main()
{
    std::cout << "I'm Going To Copy Files....\n";
    std::string src_dir{};
    std::string dst_dir{};
	std::string file_type{};
	std::cout << "Please Tell Me Where to Start:";
	std::cin >> src_dir;
	std::cout << "And Tell Me Where to Save:";
	std::cin >> dst_dir;
	std::cout << "Then Tell Me Which Type of File You Want to Copy [.jpg .bmp .png .pt]: ";
	std::cin >> file_type;
	std::string date{};
	std::cout << "Input Target Date: ";
	std::cin >> date;

	std::cout << "So, I'm Going To Copy Files From " << src_dir << " To " << dst_dir << std::endl;

	std::regex pos_pattern(R"([DU]\d(_\d)?)");  // 位置
	std::regex pattern(R"((\d{4})\.(\d{1,2})\.(\d{1,2}))");  // 日期
	// std::regex pattern(R"((\d{4})\.(\d{1,2})\.(\d{1,2}))");  // 日期
	//std::regex pattern(R"([abcde](rn))");
	std::filesystem::path src_path{ src_dir };
	for (auto& p : std::filesystem::recursive_directory_iterator(src_path))
	{
		if (p.path().has_extension() && p.path().string().find("NG") != std::string::npos && p.path().extension() == file_type) {

			std::cout << p.path().string() << std::endl;
			
			std::smatch match;
			std::string tmp{ p.path().string()};
			if (std::regex_search(tmp, match, pattern)) {
				std::cout << "-------------------" << std::endl;

				// if (match[1].str() == "1")
				// 	copy_files(p.path().string(), dst_dir + "/1");
				// else if (match[1].str() == "2")
				// 	copy_files(p.path().string(), dst_dir + "/2");
				
				if(std::stoi(match[1].str())==2025 && std::stoi(match[2].str())==4 && match[3].str()==date){

					std::smatch pm;
					std::string pos{"xxx"};
					if(std::regex_search(tmp, pm, pos_pattern)){
						pos = pm[0].str();
					}

					copy_files(p.path().string(), dst_dir, pos+"_"+p.path().filename().string());
				}
			}

		}
	}

	std::cout << "Search Complete..." << std::endl;
	return 0;

}
