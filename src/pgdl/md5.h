/*
 * @Author: laihuihang laihuihang@foxmail.com
 * @Date: 2023-06-30 15:41:48
 * @LastEditors: laihuihang laihuihang@foxmail.com
 * @LastEditTime: 2023-06-30 17:54:44
 * @FilePath: /test/md5.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef _MODEL_MD5_H_
#define _MODEL_MD5_H_

#include <string>

#define F(x,y,z) ((x & y) | (~x & z))
#define G(x,y,z) ((x & z) | (y & ~z))
#define H(x,y,z) (x^y^z)
#define I(x,y,z) (y ^ (x | ~z))
#define ROTATE_LEFT(x,n) ((x << n) | (x >> (32-n)))

#define FF(a,b,c,d,x,s,ac) \
{ \
	a += F(b,c,d) + x + ac; \
	a = ROTATE_LEFT(a,s); \
	a += b; \
}
#define GG(a,b,c,d,x,s,ac) \
{ \
	a += G(b,c,d) + x + ac; \
	a = ROTATE_LEFT(a,s); \
	a += b; \
}
#define HH(a,b,c,d,x,s,ac) \
{ \
	a += H(b,c,d) + x + ac; \
	a = ROTATE_LEFT(a,s); \
	a += b; \
}
#define II(a,b,c,d,x,s,ac) \
{ \
	a += I(b,c,d) + x + ac; \
	a = ROTATE_LEFT(a,s); \
	a += b; \
}

class MD5 
{
public:
	MD5();
	~MD5();	

	//compute file md5
	std::string ComputeFileMD5(const std::string& file_path);

	//compute string md5
	std::string ComputeStringMD5(const std::string& str);
private:
	typedef struct
	{
		unsigned int count[2];
		unsigned int state[4];
		unsigned char buffer[64];
	} MD5_CTX;

	static const int READ_DATA_SIZE = 8192;
	static const int MD5_SIZE = 16;
	static const int MD5_STR_LEN = MD5_SIZE * 2;

	void MD5Init(MD5_CTX *context);
	void MD5Update(MD5_CTX *context, const unsigned char *input, unsigned int inputlen);
	void MD5Final(MD5_CTX *context, unsigned char digest[16]);
	void MD5Transform(unsigned int state[4], const unsigned char block[64]);
	void MD5Encode(unsigned char *output, const unsigned int *input, unsigned int len);
	void MD5Decode(unsigned int *output, const unsigned char *input, unsigned int len);
};

#endif
