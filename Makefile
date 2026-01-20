###############################
##         TOOLCHAINS        ##
##       take your pick      ##
###############################

# # Normal gcc compiler (64-bit, arm64, to be used on *target*)
# CROSS_COMPILER =

# # Armel cross compiler (32-bit, arm little endian)
# CROSS_COMPILER = arm-linux-gnueabi

# # Armhf cross compiler (32-bit, arm hard float)
# CROSS_COMPILER = arm-linux-gnueabihf

# Aarch64 cross compiler (64-bit, arm64)
CROSS_COMPILER = aarch64-linux-gnu

ifneq ($(CROSS_COMPILER),)
CC      = $(CROSS_COMPILER)-gcc
OBJCOPY = $(CROSS_COMPILER)-objcopy
OBJDUMP = $(CROSS_COMPILER)-objdump
else
CC      = gcc
OBJCOPY = objcopy
OBJDUMP = objdump
endif

###############################
##        ~TOOLCHAINS        ##
###############################


CFLAGS =-Ofast -Wall -I include 
ifneq ($(CROSS_COMPILER),)
CFLAGS += -mcpu=cortex-a53+fp+simd
endif
# # If you want to use OpenMP, uncomment following CFLAG:
  CFLAGS += -fopenmp

LDFLAGS = -lm 
# # If you want to use OpenMP, uncomment following LDFLAG:
   LDFLAGS += -lgomp

SRC_DIR = src

CSRCS = $(SRC_DIR)/canny_edge.c \
        $(SRC_DIR)/hysteresis.c \
        $(SRC_DIR)/pgm_io.c
OBJS  = $(BUILD_DIR)/canny_edge.o \
        $(BUILD_DIR)/hysteresis.o \
        $(BUILD_DIR)/pgm_io.o

CSRCS_SPEEDUP = $(SRC_DIR)/canny_edge_speedup.c \
                $(SRC_DIR)/hysteresis_speedup.c \
                $(SRC_DIR)/pgm_io.c
OBJS_SPEEDUP  = $(BUILD_DIR)/canny_edge_speedup.o \
                $(BUILD_DIR)/hysteresis_speedup.o \
                $(BUILD_DIR)/pgm_io.o

CSRCS_CMP = $(SRC_DIR)/image_compare.c \
            $(SRC_DIR)/pgm_io.c
OBJS_CMP  = $(BUILD_DIR)/image_compare.o \
            $(BUILD_DIR)/pgm_io.o

########################################################
#     HOWTO ACTIVATE speedup algorithm:                #
#  $ make clean && USE_SPEEDUP=1 make run_remote
#
#  ***********
#  *IMPORTANT*:
#  ***********
#  If you enable SPEEDUP then the sources:
#     canny_edge_speedup.c and
#     hysteresis_speedup.c
#  are selected in the compilation.
#
#     HOWTO RUN BASELINE:
#  $ make clean && make run_remote
#
ifneq ($(USE_SPEEDUP),)
	CFLAGS += -DSPEEDUP
endif
#
#     ~HOWTO ACTIVATE speedup                          #
########################################################


########################################################
#     HOWTO ACTIVATE verbose mode:                     #
#  $ make clean && USE_VERBOSE=1 make run_remote
#
ifneq ($(USE_VERBOSE),)
	CFLAGS += -DVERBOSE=1
endif
#     ~HOWTO ACTIVATE verbose mode                     #
########################################################


HOME_BEAGLE          = /home/beagle
BENCHMARK            = canny-edge
BENCHMARK_DIR        = $(HOME)/bench/$(BENCHMARK)
BENCHMARK_DIR_REMOTE = $(HOME_BEAGLE)/bench/$(BENCHMARK)

IMG_DIR = img

########################################################
#
# Choose input image: use one of the following pictures for testing
#    Uncomment the one you want, comment the others
#
# IMG = $(IMG_DIR)/klomp.pgm
# IMG = $(IMG_DIR)/square.pgm
# IMG = $(IMG_DIR)/tiger.pgm
  IMG = $(IMG_DIR)/Dennis.pgm
#
#
#
# Below conditionals determine the arguments for the image compare program.
# If you add your own image(s) to this Makefile, these conditionals should
# be extended!
ifeq ($(IMG),$(IMG_DIR)/klomp.pgm)
	IMG_CMP1=$(IMG_DIR)/klomp-edge-baseline.pgm
	IMG_CMP2=$(BUILD_DIR)/img/klomp.pgm_s_2.50_l_0.50_h_0.50.pgm
else ifeq ($(IMG),$(IMG_DIR)/square.pgm)
	IMG_CMP1=$(IMG_DIR)/square-edge-baseline.pgm
	IMG_CMP2=$(BUILD_DIR)/img/square.pgm_s_2.50_l_0.50_h_0.50.pgm
else ifeq ($(IMG),$(IMG_DIR)/tiger.pgm)
	IMG_CMP1=$(IMG_DIR)/tiger-edge-baseline.pgm
	IMG_CMP2=$(BUILD_DIR)/img/tiger.pgm_s_2.50_l_0.50_h_0.50.pgm
else ifeq ($(IMG),$(IMG_DIR)/Dennis.pgm)
	IMG_CMP1=$(IMG_DIR)/Dennis-edge-baseline.pgm
	IMG_CMP2=$(BUILD_DIR)/img/Dennis.pgm_s_2.50_l_0.50_h_0.50.pgm
endif
#
# ~Choose input image
########################################################



BUILD_DIR     = build

# This is where the resulting output images are written to:
BUILD_IMG_DIR = build/img

TARGET         = $(BUILD_DIR)/$(BENCHMARK)
TARGET_SPEEDUP = $(BUILD_DIR)/$(BENCHMARK)-speedup
TARGET_CMP     = $(BUILD_DIR)/image-compare

DISASSEMBLY         = $(TARGET)-disassembly.txt
DISASSEMBLY_SPEEDUP = $(TARGET_SPEEDUP)-disassembly.txt

CMD        = $(TARGET) $(IMG)
CMD_REMOTE = ssh beagley "cd $(BENCHMARK_DIR_REMOTE) && $(TARGET) $(IMG)"

CMD_SPEEDUP        = $(TARGET_SPEEDUP) $(IMG)
CMD_SPEEDUP_REMOTE = ssh beagley "cd $(BENCHMARK_DIR_REMOTE) && \
	$(TARGET_SPEEDUP) $(IMG)"

CMD_CMP        = $(TARGET_CMP) $(IMG_CMP1) $(IMG_CMP2)
CMD_CMP_REMOTE = ssh beagley "cd $(BENCHMARK_DIR_REMOTE) && \
	$(TARGET_CMP) $(IMG_CMP1) $(IMG_CMP2)"


.PHONY: default run run_remote compare compare_remote copy2remote clean bear

default: $(TARGET) $(TARGET_SPEEDUP) $(TARGET_CMP) \
	$(DISASSEMBLY) $(DISASSEMBLY_SPEEDUP)

$(TARGET): $(OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

$(DISASSEMBLY): $(TARGET)
	$(OBJDUMP) -d -h -p $^ > $@

$(TARGET_SPEEDUP): $(OBJS_SPEEDUP)
	$(CC) $^ -o $@ $(LDFLAGS)

$(DISASSEMBLY_SPEEDUP): $(TARGET_SPEEDUP)
	$(OBJDUMP) -d -h -p $^ > $@

$(TARGET_CMP): $(OBJS_CMP)
	$(CC) $^ -o $@

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) -c $(CFLAGS) -o $@ $<
-include $(OBJS:.o=.d)
-include $(OBJS_SPEEDUP:.o=.d)
-include $(OBJS_CMP:.o=.d)

# Run the target locally
ifneq ($(USE_SPEEDUP),)
run: $(TARGET_SPEEDUP)
	$(CMD_SPEEDUP)
else
run: $(TARGET)
	$(CMD)
endif

# Run the target remotely
ifneq ($(USE_SPEEDUP),)
run_remote: $(TARGET_SPEEDUP) copy2remote
	$(CMD_SPEEDUP_REMOTE)
else
run_remote: $(TARGET) copy2remote
	$(CMD_REMOTE)
endif

# Compare the resulting image
compare: $(TARGET_CMP)
	$(CMD_CMP)

# Compare the resulting image on the remote target
compare_remote: $(TARGET_CMP)
	$(CMD_CMP_REMOTE)

# Copy necessary files to the remote target
ifneq ($(USE_SPEEDUP),)
copy2remote: $(TARGET_SPEEDUP) $(TARGET_CMP)
	ssh beagley "mkdir -p $(BENCHMARK_DIR_REMOTE)/$(BUILD_IMG_DIR)"
	rsync -av $(IMG_DIR)/*.pgm beagley:$(BENCHMARK_DIR_REMOTE)/$(IMG_DIR)/
	scp -p $(TARGET_SPEEDUP) beagley:$(BENCHMARK_DIR_REMOTE)/$(BUILD_DIR)
	scp -p $(TARGET_CMP) beagley:$(BENCHMARK_DIR_REMOTE)/$(BUILD_DIR)
else
copy2remote: $(TARGET) $(TARGET_CMP)
	ssh beagley "mkdir -p $(BENCHMARK_DIR_REMOTE)/$(BUILD_IMG_DIR)"
	rsync -av $(IMG_DIR)/*.pgm beagley:$(BENCHMARK_DIR_REMOTE)/$(IMG_DIR)/
	scp -p $(TARGET) beagley:$(BENCHMARK_DIR_REMOTE)/$(BUILD_DIR)
	scp -p $(TARGET_CMP) beagley:$(BENCHMARK_DIR_REMOTE)/$(BUILD_DIR)
endif

clean:
	@rm -rf $(BUILD_DIR)/*.o $(TARGET) $(TARGET_SPEEDUP) $(TARGET_CMP) \
		$(DISASSEMBLY) $(DISASSEMBLY_SPEEDUP)


# Bear: create compile_commands.json for clangd running in the background
# (note: bear must be installed for this to work)
bear: clean
	bear -- make
