#ifndef GMMU_H_
#define GMMU_H_

#include <map>
#include <set>
#include <list>
#include <functional>
#include <unordered_set>
#include "../cuda-sim/memory.h"
#include "gpu-sim.h"

class gpgpu_sim_config;

// uvm related stats: originaly in gpgpu-sim.h
extern unsigned long long kernel_time;
extern unsigned long long memory_copy_time_h2d;
extern unsigned long long memory_copy_time_d2h;
extern unsigned long long prefetch_time;
extern unsigned long long devicesync_time;
extern unsigned long long writeback_time;
extern unsigned long long dma_time;

enum stats_type {
  prefetch = 0,
  prefetch_breakdown,
  memcpy_h2d,
  memcpy_d2h,
  memcpy_d2d,
  kernel_launch,
  page_fault,
  device_sync,
  write_back,
  invalidate,
  dma
};
// uvm related stats

class event_stats {
public:
  event_stats(enum stats_type t, unsigned long long s_time,
              unsigned long long e_time)
      : type(t), start_time(s_time), end_time(e_time) {}
  event_stats(enum stats_type t, unsigned long long s_time)
      : type(t), start_time(s_time), end_time(0) {}
  enum stats_type type;
  unsigned long long start_time;
  unsigned long long end_time;

  virtual void print(FILE *fout, float freq) = 0;
  virtual void calculate() = 0;
};

class memory_stats : public event_stats {
public:
  memory_stats(enum stats_type t, unsigned long long s_time, mem_addr_t s_addr,
               size_t sz, unsigned s_id)
      : event_stats(t, s_time), start_addr(s_addr), size(sz), stream_id(s_id) {}
  memory_stats(enum stats_type t, unsigned long long s_time,
               unsigned long long e_time, mem_addr_t s_addr, size_t sz,
               unsigned s_id)
      : event_stats(t, s_time, e_time), start_addr(s_addr), size(sz),
        stream_id(s_id) {}
  mem_addr_t start_addr;
  size_t size;
  unsigned stream_id;

  virtual void print(FILE *fout, float freq) {
    fprintf(fout, "F: %8llu----T: %8llu \t St: %x Sz: %lu \t Sm: %u \t ",
            start_time, end_time, start_addr, size, stream_id);
    if (type == memcpy_h2d)
      fprintf(fout, "T: memcpy_h2d");
    else if (type == memcpy_d2h)
      fprintf(fout, "T: memcpy_d2h");
    else if (type == memcpy_d2d)
      fprintf(fout, "T: memcpy_d2d");
    else if (type == prefetch)
      fprintf(fout, "T: prefetch");
    else if (type == prefetch_breakdown)
      fprintf(fout, "T: prefetch_breakdown");
    else if (type == device_sync)
      fprintf(fout, "T: device_sync");
    else if (type == write_back)
      fprintf(fout, "T: write_back");
    else if (type == invalidate)
      fprintf(fout, "T: invalidate");
    else if (type == dma)
      fprintf(fout, "T: dma");

    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }
  virtual void calculate() {
    if (type == memcpy_h2d) {
      memory_copy_time_h2d += end_time - start_time;
    } else if (type == memcpy_d2h) {
      memory_copy_time_d2h += end_time - start_time;
    } else if (type == prefetch_breakdown) {
      prefetch_time += end_time - start_time;
    } else if (type == device_sync) {
      devicesync_time += end_time - start_time;
    } else if (type == write_back) {
      writeback_time += end_time - start_time;
    } else if (type == dma) {
      dma_time += end_time - start_time;
    }
  }
};

class kernel_stats : public event_stats {
public:
  kernel_stats(unsigned long long s_time, unsigned s_id, unsigned k_id)
      : event_stats(kernel_launch, s_time), stream_id(s_id), kernel_id(k_id) {}
  unsigned stream_id;
  unsigned kernel_id;

  virtual void print(FILE *fout, float freq) {
    fprintf(
        fout,
        "F: %8llu----T: %8llu \t \t \t Kl: %u \t Sm: %u \t T: kernel_launch",
        start_time, end_time, kernel_id, stream_id);
    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }

  virtual void calculate() { kernel_time += end_time - start_time; }
};

class page_fault_stats : public event_stats {
public:
  page_fault_stats(unsigned long long s_time, const std::list<mem_addr_t> &pgs,
                   unsigned sz)
      : event_stats(page_fault, s_time), pages(pgs), transfering_pages(pgs),
        size(sz) {}
  std::list<mem_addr_t> pages;
  std::list<mem_addr_t> transfering_pages;
  size_t size;

  virtual void print(FILE *fout, float freq) {
    fprintf(fout, "F: %8llu----T: %8llu \t Sz: %lu \t T: page_fault",
            start_time, end_time, size);
    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }

  virtual void calculate() {}
};

// In order to skip idle cycles due to page fault, we need collect warp info
// in all cores.
extern std::list<shd_warp_t *> all_warps;
extern std::list<shd_warp_t *> fail_warps;
extern bool skip_cycles;
extern bool skip_cycles_enable;

extern std::map<unsigned long long, std::list<event_stats *>> sim_prof;

extern bool sim_prof_enable;


void print_sim_prof(FILE *fout, float freq);

void calculate_sim_prof(FILE *fout, gpgpu_sim *gpu);

void update_sim_prof_kernel(unsigned kernel_id, unsigned long long end_time);

void update_sim_prof_prefetch(mem_addr_t start_addr, size_t size,
                              unsigned long long end_time);

void update_sim_prof_prefetch_break_down(unsigned long long end_time);

void print_UVM_stats(gpgpu_new_stats *new_stats, gpgpu_sim *gpu, FILE *fout);

class access_info {
public:
  mem_addr_t page_no;
  mem_addr_t mem_addr;
  size_t size;
  unsigned long long cycle;
  bool is_read;
  unsigned sm_id;
  unsigned warp_id;
  access_info(mem_addr_t p_n, mem_addr_t addr, size_t s, unsigned long long c,
              bool rw, unsigned s_id, unsigned w_id)
      : page_no(p_n), mem_addr(addr), size(s), cycle(c), is_read(rw),
        sm_id(s_id), warp_id(w_id) {}
};

class gpgpu_new_stats {
public:
  gpgpu_new_stats(const gpgpu_sim_config &config);
  ~gpgpu_new_stats();
  void print(FILE *fout) const;
  void print_pcie(FILE *fout) const;
  void print_access_pattern_detail(FILE *fout) const;

  // Figure 1, 6 ok
  // for each page, tot cnt, tlb, pf, etc.
  void print_access_pattern(FILE *fout);

  // Figure 2 ok
  // for each page access, which cycle, shader, etc
  void print_time_and_access(FILE *fout) const;

  // Figure 3, 5 ok
  // cycles how pf and pe numbers change
  void print_cycle_pf_nums(FILE *fout);
  // Figure 3, 5 ok
  void print_cycle_pe_nums(FILE *fout);

  // Figure 4
  void print_cycle_pageset(FILE*fout);

  // Figure 7
  // for all page access, tlb, pf, etc.
  void print_page_access_stats (FILE *fout) const;

  // Figure 9
  // for accessed pages, read and write portion
  void print_page_read_write(FILE *fout);

  
  // tlb hit
  unsigned long long *tlb_hit;
  // tlb miss
  unsigned long long *tlb_miss;

  // tlb validate
  unsigned long long *tlb_val;
  // tlb eviction
  unsigned long long *tlb_evict;
  // tlb invalidated by page eviction
  unsigned long long *tlb_page_evict;

  // in tlb miss, page hit
  unsigned long long *mf_page_hit;
  // in tlb miss, page miss
  unsigned long long *mf_page_miss;

  // in tlb miss, page miss, the first create fault
  unsigned long long mf_page_fault_outstanding;
  // in tlb miss, page miss, the following that appends to mshr
  // maybe the same to previous fault
  unsigned long long mf_page_fault_pending;

  unsigned long long page_evict_dirty;

  unsigned long long page_evict_not_dirty;

  // prefetch page hit
  unsigned long long pf_page_hit;
  // prefetch page miss
  unsigned long long pf_page_miss;
  // prefetch fault page size, large page and latency
  std::vector<std::pair<unsigned long, unsigned long long>> pf_fault_latency;

  // for each page, how many time is it being accessed by each shader
  std::map<mem_addr_t, unsigned> *page_access_times;

  // how mant pf this page suffered when access
  // pfn -> ctr
  // this is not per shader, overall
  // just on this pfn from whatever shader
  // tlb miss pt miss
  // ok
  std::map<mem_addr_t, unsigned> page_fault_outstand_times; 
  // ok
  std::map<mem_addr_t, unsigned> page_fault_pending_times;
  // ok
  std::map<mem_addr_t, unsigned> page_tlb_miss_pt_miss_times;
  // ok
  std::map<mem_addr_t, unsigned> page_tlb_miss_pt_hit_times;
  // ok
  std::map<mem_addr_t, unsigned> page_tlb_hit_times;
  // ok
  std::map<mem_addr_t, unsigned> page_tlb_miss_times;

  // ok
  // first : read times
  // secind: write times
  std::map<mem_addr_t, std::pair<unsigned,unsigned>> page_rw_times;
  // // ok
  // std::map<mem_addr_t, unsigned> page_write_times;
  // // ok
  // std::map<mem_addr_t, bool> page_suffer_read;
  // // ok
  // std::map<mem_addr_t, bool> page_suffer_write;
  
  // ok outstanding + pending
  // scale to 1000 cyles frame unit
  std::map<unsigned long long, std::vector<mem_addr_t>> cycle_pf_times;
  

  // scale to 1000 cyles frame unit
  std::map<unsigned long long, std::vector<mem_addr_t>> cycle_pe_times;

  // scale to 1000 cycles 
  // how many pages are in current global memory
  // scan the page_table (or the valid page list)
  std::map<unsigned long long, std::set<mem_addr_t>> cycle_pages_valid;

  // for each timestamp, which page is being accessed
  std::list<access_info> time_and_page_access;

  // ready lanes utilization
  std::list<std::pair<unsigned long long, float>> pcie_read_utilization;
  // write lanes utilization
  std::list<std::pair<unsigned long long, float>> pcie_write_utilization;

  // page and its partern
  std::map<mem_addr_t, std::vector<bool>> page_thrashing;
  // tlb and its partern
  std::map<mem_addr_t, std::vector<bool>> *tlb_thrashing;

  // for each shader, the memory access latency
  std::map<unsigned, std::pair<bool, unsigned long long>> * ma_latency;

  // for mf when it is fault(not pending to prefetch), the latency
  std::map<mem_addr_t, std::list<unsigned long long>> mf_page_fault_latency;

  // for prefetch each small page latency
  std::map<mem_addr_t, std::list<unsigned long long>> pf_page_fault_latency;

  const gpgpu_sim_config &m_config;

  unsigned long long num_dma;
  unsigned long long dma_page_transfer_read;
  unsigned long long dma_page_transfer_write;
};

// this class simulate the gmmu unit on chip
class gmmu_t {
public:
  gmmu_t(class gpgpu_sim *gpu, const gpgpu_sim_config &config,
         class gpgpu_new_stats *new_stats);
  unsigned long long calculate_transfer_time(size_t data_size);
  void calculate_devicesync_time(size_t data_size);
  void cycle();
  void register_tlbflush_callback(std::function<void(mem_addr_t)> cb_tlb);
  void tlb_flush(mem_addr_t page_num);
  void page_eviction_procedure();
  bool is_block_evictable(mem_addr_t bb_addr, size_t size);

  // add a new accessed page or refresh the position of the page in the LRU page
  // list being called on detecting tlb hit or when memory fetch comes back from
  // the upward (gmmu to cu) queue
  void refresh_valid_pages(mem_addr_t page_addr);
  void sort_valid_pages();

  // check whether the page to be accessed is already in pci-e write stage queue
  // being called on tlb hit or on tlb miss but no page fault
  void check_write_stage_queue(mem_addr_t page_num, bool refresh);

  void valid_pages_erase(mem_addr_t pagenum);
  void valid_pages_clear();

  void register_prefetch(mem_addr_t m_device_addr,mem_addr_t m_device_allocation_ptr, size_t m_cnt,struct CUstream_st *m_stream);
  void activate_prefetch(mem_addr_t m_device_addr, size_t m_cnt, struct CUstream_st *m_stream);

  struct lp_tree_node *build_lp_tree(mem_addr_t addr, size_t size);
  void reset_large_page_info(struct lp_tree_node *node);
  void reset_lp_tree_node(struct lp_tree_node *node);
  struct lp_tree_node *get_lp_node(mem_addr_t addr);
  void evict_whole_tree(struct lp_tree_node *root);
  mem_addr_t update_basic_block(struct lp_tree_node *root, mem_addr_t addr, size_t size, bool prefetch);
  mem_addr_t get_basic_block(struct lp_tree_node *root, mem_addr_t addr);

  void fill_lp_tree(struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks);
  void remove_lp_tree(struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks);
  void traverse_and_fill_lp_tree(struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks);
  void traverse_and_remove_lp_tree(struct lp_tree_node *node,std::set<mem_addr_t> &scheduled_basic_blocks);

  bool pcie_transfers_completed();

  void initialize_large_page(mem_addr_t start_addr, size_t size);

  int get_trng(int low, int high);
  

  unsigned long long get_ready_cycle_dma(unsigned size);

  float get_pcie_utilization(unsigned num_pages);

  void do_hardware_prefetch(std::map<mem_addr_t, std::list<mem_fetch *>> &page_fault_this_turn);

  void reserve_pages_insert(mem_addr_t addr, unsigned mem_access_uid);
  void reserve_pages_remove(mem_addr_t addr, unsigned mem_access_uid);
  bool reserve_pages_check(mem_addr_t addr);

  std::map<mem_addr_t, std::list<unsigned>> reserve_pages;

  void update_hardware_prefetcher_oversubscribed();

  // update paging, pinning, and eviction decision based on memory access
  // pattern under oversubscription
  void update_memory_management_policy();
  void log_kernel_info(unsigned kernel_id, unsigned long long time, bool finish);

  void reset_large_page_info();

  mem_addr_t get_eviction_base_addr(mem_addr_t page_addr);
  size_t get_eviction_granularity(mem_addr_t page_addr);

  int get_bb_access_counter(struct lp_tree_node *node, mem_addr_t addr);
  int get_bb_round_trip(struct lp_tree_node *node, mem_addr_t addr);
  void inc_bb_access_counter(mem_addr_t addr);
  void inc_bb_round_trip(struct lp_tree_node *root);
  void traverse_and_reset_access_counter(struct lp_tree_node *root);
  void reset_bb_access_counter();
  void traverse_and_reset_round_trip(struct lp_tree_node *root);
  void reset_bb_round_trip();
  void update_access_type(mem_addr_t addr, int type);

  bool should_cause_page_migration(mem_addr_t addr, bool is_write);

  // Ramulator & UVM:
  // validate queue
  void push_pcie_gmmu_queue(mem_addr_t page_num) {
    m_pcie_gmmu_queue.push_back(page_num);
  }
  bool is_pcie_gmmu_queue_empty() {
    return m_pcie_gmmu_queue.empty();
  }
  mem_addr_t front_pcie_gmmu_queue() {
    assert(!is_pcie_gmmu_queue_empty());
    return m_pcie_gmmu_queue.front();
  }
  void pop_pcie_gmmu_queue() {
    assert(!is_pcie_gmmu_queue_empty());
    m_pcie_gmmu_queue.pop_front();
  }

  // invalidate queue
  void push_pcie_gmmu_invalidate_queue(mem_addr_t page_num) {
    m_pcie_gmmu_invalidate_queue.push_back(page_num);
  }
  bool is_pcie_gmmu_invalidate_queue_empty() {
    return m_pcie_gmmu_invalidate_queue.empty();
  }
  mem_addr_t front_pcie_gmmu_invalidate_queue() {
    assert(!is_pcie_gmmu_invalidate_queue_empty());
    return m_pcie_gmmu_invalidate_queue.front();
  }
  void pop_pcie_gmmu_invalidate_queue() {
    assert(!is_pcie_gmmu_invalidate_queue_empty());
    m_pcie_gmmu_invalidate_queue.pop_front();
  }
  // Ramulator & UVM


  // UVM:
  // used for uvm on dram system
  // not related the cpu-gpu uvm
  std::list<mem_addr_t> m_pcie_gmmu_queue;
  std::list<mem_addr_t> m_pcie_gmmu_invalidate_queue;
  // Ramulator & UVM

  // data structure to wrap memory fetch and page table walk delay
  struct page_table_walk_latency_t {
    mem_fetch *mf;
    unsigned long long ready_cycle;
  };

  // page table walk delay queue
  std::list<page_table_walk_latency_t> page_table_walk_queue;

  enum class latency_type {
    PCIE_READ,
    // for evitction
    PCIE_WRITE_BACK,
    INVALIDATE,
    PAGE_FAULT,
    DMA
  };

  // data structure to wrap a memory page and delay to transfer over PCI-E
  struct pcie_latency_t {
    // start page addressor PFN?
    mem_addr_t start_addr;
    // a set of pages connected (a block)
    unsigned long long size;
    // bind with a list of pages
    // this address is PFN
    // continous pages
    std::list<mem_addr_t> page_list;
    // when this transaction will be ready
    unsigned long long ready_cycle;

    mem_fetch *mf;
    latency_type type;
  };

  // staging queue to hold the PCI-E requests waiting for scheduling
  std::list<pcie_latency_t *> pcie_read_stage_queue;
  std::list<pcie_latency_t *> pcie_write_stage_queue;

  // read queue for fetching the page from host side
  // the request may be global memory's read (load)/ write (store)
  pcie_latency_t *pcie_read_latency_queue;

  // write back queue for page eviction requests over PCI-E
  pcie_latency_t *pcie_write_latency_queue;

  unsigned long long get_ready_cycle(pcie_latency_t * p_pcie_latency_queue);
  unsigned long long get_ready_cycle(unsigned num_pages);

  // loosely represent MSHRs to hold all memory fetches
  // corresponding to a PCI-E read requests, i.e., a common page number
  // to replay the memory fetch back upon completion
  std::map<mem_addr_t, std::list<mem_fetch *>> req_info;

  // need the gpu to do address traslation, validate page
  class gpgpu_sim *m_gpu;

  // config file
  const gpgpu_sim_config &m_config;
  const struct shader_core_config *m_shader_config;

  // callback functions to invalidate the tlb in ldst unit
  std::list<std::function<void(mem_addr_t)>> callback_tlb_flush;

  // list of valid pages (valid = 1, accessed = 1/0, dirty = 1/0) ordered as LRU
  std::list<eviction_t *> valid_pages;

  // page eviction policy
  enum class eviction_policy { LRU, TBN, SEQUENTIAL_LOCAL, RANDOM, LFU, LRU4K };

  // types of hardware prefetcher
  enum class hwardware_prefetcher { DISBALED, TBN, SEQUENTIAL_LOCAL, RANDOM };

  // types of hardware prefetcher under over-subscription
  enum class hwardware_prefetcher_oversub {
    DISBALED,
    TBN,
    SEQUENTIAL_LOCAL,
    RANDOM
  };

  // type of DMA
  enum class dma_type { DISABLED, ADAPTIVE, ALWAYS, OVERSUB };

  // type of memory access pattern per data structure
  enum class ds_pattern {
    UNDECIDED,
    RANDOM,
    LINEAR,
    MIXED,
    RANDOM_REUSE,
    LINEAR_REUSE,
    MIXED_REUSE
  };

  // list of scheduled basic blocks by their timestamps
  std::list<std::pair<unsigned long long, mem_addr_t>> block_access_list;

  // list of launch and finish cycle of kernels keyed by id
  std::map<unsigned, std::pair<unsigned long long, unsigned long long>>
      kernel_info;

  // for the best performance
  // TBN should be used for both eviction and prefetching
  eviction_policy evict_policy;
  hwardware_prefetcher prefetcher;
  hwardware_prefetcher_oversub oversub_prefetcher;

  dma_type dma_mode;

  struct prefetch_req {
    // starting address (rolled up and down for page alignment) for the prefetch
    mem_addr_t start_addr;

    // current address from the start up to which PCI-e has already processed
    mem_addr_t cur_addr;

    // starting address of the current variable allocation
    mem_addr_t allocation_addr;

    // total size (rolled up and down for page alignment) for the prefetch
    size_t size;

    // stream associated to the prefetch
    CUstream_st *m_stream;

    // memory fetches, which are created upon page fault and are depending on
    // current prefetch, aggreagted before the prefetch is actually scheduled
    std::map<mem_addr_t, std::list<mem_fetch *>> incoming_replayable_nacks;

    // memory fetches that are finished PCI-e transfer are aggregated to be
    // replayed together upon completion of the prefetch
    std::map<mem_addr_t, std::list<mem_fetch *>> outgoing_replayable_nacks;

    // list of pages (max upto 2MB) from the current prefetch request which are
    // being served by PCI-e
    std::list<mem_addr_t> pending_prefetch;

    // stream manager upon reaching to this entry of the queue sets it to active
    bool active;
  };

  std::list<prefetch_req> prefetch_req_buffer;

  std::list<event_stats *> fault_stats;
  std::list<event_stats *> writeback_stats;

  std::list<struct lp_tree_node *> large_page_info;
  size_t total_allocation_size;

  bool over_sub;

  class gpgpu_new_stats *m_new_stats;
};

// the binary tree in uvm module
struct lp_tree_node {
  mem_addr_t addr;
  // exact size holding pages
  size_t size;
  size_t valid_size;
  struct lp_tree_node *left;
  struct lp_tree_node *right;
  uint32_t access_counter;
  uint8_t RW;
};


#endif
