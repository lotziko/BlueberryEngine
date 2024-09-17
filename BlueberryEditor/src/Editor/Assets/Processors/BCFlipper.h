#pragma once

// Based on https://github.com/inequation/flip-s3tc/blob/master/flip-s3tc.h

#include <stdint.h>

namespace Blueberry
{
	struct BC1Block
	{
		uint16_t c0, c1;
		uint8_t dcba, hgfe, lkji, ponm;
	};

	struct BC3Block
	{
		uint8_t a0, a1;
		struct
		{
			uint8_t	alpha[3];
		} ahagafaeadacabaa, apaoanamalakajai;
		uint16_t c0, c1;
		uint8_t dcba, hgfe, lkji, ponm;
	};

	struct BC4Block
	{
		uint8_t r0, r1;
		struct
		{
			uint8_t	red[3];
		} rhrgrfrerdrcrbra,	rprornrmrlrkrjri;
	};

	struct BC5Block
	{
		uint8_t r0, r1;
		struct
		{
			uint8_t	red[3];
		} rhrgrfrerdrcrbra, rprornrmrlrkrjri;
		uint8_t g0, g1;
		struct
		{
			uint8_t	green[3];
		} ghgggfgegdgcgbga, gpgogngmglgkgjgi;
	};

	/** Performs an Y-flip of the given BC1 block in place. */
	void FlipBC1Block(struct BC1Block* block)
	{
		uint8_t temp;

		temp = block->dcba;
		block->dcba = block->ponm;
		block->ponm = temp;
		temp = block->hgfe;
		block->hgfe = block->lkji;
		block->lkji = temp;
	}

	/** Performs an Y-flip of the given BC3 block in place. */
	void FlipBC3Block(struct BC3Block* block)
	{
		uint8_t temp8;
		uint32_t temp32;
		uint32_t *asInt[2];

		asInt[0] = (uint32_t *)block->ahagafaeadacabaa.alpha;
		asInt[1] = (uint32_t *)block->apaoanamalakajai.alpha;
		// swap adacabaa with apaoanam
		temp32 = *asInt[0] & ((1 << 12) - 1);
		*asInt[0] &= ~((1 << 12) - 1);
		*asInt[0] |= (*asInt[1] & (((1 << 12) - 1) << 12)) >> 12;
		*asInt[1] &= ~(((1 << 12) - 1) << 12);
		*asInt[1] |= temp32 << 12;
		// swap ahagafae with alakajai
		temp32 = *asInt[0] & (((1 << 12) - 1) << 12);
		*asInt[0] &= ~(((1 << 12) - 1) << 12);
		*asInt[0] |= (*asInt[1] & ((1 << 12) - 1)) << 12;
		*asInt[1] &= ~((1 << 12) - 1);
		*asInt[1] |= temp32 >> 12;

		temp8 = block->dcba;
		block->dcba = block->ponm;
		block->ponm = temp8;
		temp8 = block->hgfe;
		block->hgfe = block->lkji;
		block->lkji = temp8;
	}

	/** Performs an Y-flip of the given BC4 block in place. */
	void FlipBC4Block(struct BC4Block *block)
	{
		uint32_t temp32;
		uint32_t *asInt[2];

		asInt[0] = (uint32_t*)block->rhrgrfrerdrcrbra.red;
		asInt[1] = (uint32_t*)block->rprornrmrlrkrjri.red;

		// swap rdrcrbra with rprornrm
		temp32 = *asInt[0] & ((1 << 12) - 1);
		*asInt[0] &= ~((1 << 12) - 1);
		*asInt[0] |= (*asInt[1] & (((1 << 12) - 1) << 12)) >> 12;
		*asInt[1] &= ~(((1 << 12) - 1) << 12);
		*asInt[1] |= temp32 << 12;
		// swap rhrgrfre with rlrkrjri
		temp32 = *asInt[0] & (((1 << 12) - 1) << 12);
		*asInt[0] &= ~(((1 << 12) - 1) << 12);
		*asInt[0] |= (*asInt[1] & ((1 << 12) - 1)) << 12;
		*asInt[1] &= ~((1 << 12) - 1);
		*asInt[1] |= temp32 >> 12;
	}

	/** Performs an Y-flip of the given BC5 block in place. */
	void FlipBC5Block(struct BC5Block* block)
	{
		uint32_t temp32;
		uint32_t *asInt[2];

		asInt[0] = (uint32_t*)block->rhrgrfrerdrcrbra.red;
		asInt[1] = (uint32_t*)block->rprornrmrlrkrjri.red;

		// swap rdrcrbra with rprornrm
		temp32 = *asInt[0] & ((1 << 12) - 1);
		*asInt[0] &= ~((1 << 12) - 1);
		*asInt[0] |= (*asInt[1] & (((1 << 12) - 1) << 12)) >> 12;
		*asInt[1] &= ~(((1 << 12) - 1) << 12);
		*asInt[1] |= temp32 << 12;
		// swap rhrgrfre with rlrkrjri
		temp32 = *asInt[0] & (((1 << 12) - 1) << 12);
		*asInt[0] &= ~(((1 << 12) - 1) << 12);
		*asInt[0] |= (*asInt[1] & ((1 << 12) - 1)) << 12;
		*asInt[1] &= ~((1 << 12) - 1);
		*asInt[1] |= temp32 >> 12;

		asInt[0] = (uint32_t*)block->ghgggfgegdgcgbga.green;
		asInt[1] = (uint32_t*)block->gpgogngmglgkgjgi.green;

		// swap gdgcgbga with gpgogngm
		temp32 = *asInt[0] & ((1 << 12) - 1);
		*asInt[0] &= ~((1 << 12) - 1);
		*asInt[0] |= (*asInt[1] & (((1 << 12) - 1) << 12)) >> 12;
		*asInt[1] &= ~(((1 << 12) - 1) << 12);
		*asInt[1] |= temp32 << 12;
		// swap ghgggfge with glgkgjgi
		temp32 = *asInt[0] & (((1 << 12) - 1) << 12);
		*asInt[0] &= ~(((1 << 12) - 1) << 12);
		*asInt[0] |= (*asInt[1] & ((1 << 12) - 1)) << 12;
		*asInt[1] &= ~((1 << 12) - 1);
		*asInt[1] |= temp32 >> 12;
	}

	/**
	 * Performs an Y-flip on the given DXT1 image in place.
	 * @param	data	buffer with image data
	 * @param	width	image width in pixels
	 * @param	height	image height in pixels
	 */
	void FlipBC1Image(void *data, int width, int height)
	{
		int x, y;
		struct BC1Block temp1, temp2;
		struct BC1Block* blocks = (struct BC1Block*)data;

		width = (width + 3) / 4;
		height = (height + 3) / 4;

		for (y = 0; y < height / 2; ++y)
		{
			for (x = 0; x < width; ++x)
			{
				temp1 = blocks[y * width + x];
				temp2 = blocks[(height - y - 1) * width + x];
				FlipBC1Block(&temp1);
				FlipBC1Block(&temp2);
				blocks[(height - y - 1) * width + x] = temp1;
				blocks[y * width + x] = temp2;
			}
		}
	}

	/**
	 * Performs an Y-flip on the given BC3 image in place.
	 * @param	data	buffer with image data
	 * @param	width	image width in pixels
	 * @param	height	image height in pixels
	 */
	void FlipBC3Image(void *data, int width, int height)
	{
		int x, y;
		struct BC3Block temp1, temp2;
		struct BC3Block* blocks = (struct BC3Block*)data;

		width = (width + 3) / 4;
		height = (height + 3) / 4;

		for (y = 0; y < height / 2; ++y)
		{
			for (x = 0; x < width; ++x)
			{
				temp1 = blocks[y * width + x];
				temp2 = blocks[(height - y - 1) * width + x];
				FlipBC3Block(&temp1);
				FlipBC3Block(&temp2);
				blocks[(height - y - 1) * width + x] = temp1;
				blocks[y * width + x] = temp2;
			}
		}
	}

	/**
	 * Performs an Y-flip on the given BC4 image in place.
	 * @param	data	buffer with image data
	 * @param	width	image width in pixels
	 * @param	height	image height in pixels
	 */
	void FlipBC4Image(void *data, int width, int height)
	{
		int x, y;
		struct BC4Block temp1, temp2;
		struct BC4Block* blocks = (struct BC4Block*)data;

		width = (width + 3) / 4;
		height = (height + 3) / 4;

		for (y = 0; y < height / 2; ++y)
		{
			for (x = 0; x < width; ++x)
			{
				temp1 = blocks[y * width + x];
				temp2 = blocks[(height - y - 1) * width + x];
				FlipBC4Block(&temp1);
				FlipBC4Block(&temp2);
				blocks[(height - y - 1) * width + x] = temp1;
				blocks[y * width + x] = temp2;
			}
		}
	}

	/**
	 * Performs an Y-flip on the given BC5 image in place.
	 * @param	data	buffer with image data
	 * @param	width	image width in pixels
	 * @param	height	image height in pixels
	 */
	void FlipBC5Image(void *data, int width, int height)
	{
		int x, y;
		struct BC5Block temp1, temp2;
		struct BC5Block* blocks = (struct BC5Block*)data;

		width = (width + 3) / 4;
		height = (height + 3) / 4;

		for (y = 0; y < height / 2; ++y)
		{
			for (x = 0; x < width; ++x)
			{
				temp1 = blocks[y * width + x];
				temp2 = blocks[(height - y - 1) * width + x];
				FlipBC5Block(&temp1);
				FlipBC5Block(&temp2);
				blocks[(height - y - 1) * width + x] = temp1;
				blocks[y * width + x] = temp2;
			}
		}
	}
}