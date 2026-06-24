#include "gmmu.h"
#include "l2cache.h"
#include "../abstract_hardware_model.h"

#include <bitset>
#include <list>
#include <iostream>

extern bool skip_cycles;
extern bool skip_cycles_enable;
extern int skipped_cycles;
extern int gpu_tot_skipped_cycle;
extern int g_debug_execution;
extern int new_debug_execution;
extern unsigned long long acc_page_cnt;

void print_sim_prof(FILE *fout, float freq) {
  freq /= 1000;
  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      (*iter2)->print(fout, freq);
    }
  }
}

// only called at the exit
void print_UVM_stats(gpgpu_new_stats *new_stats, gpgpu_sim *gpu, FILE *fout) {
  new_stats->print(stdout);

  /*
      FILE* f1 = fopen("Pcie_trace.txt", "w");

      g_the_gpu->m_new_stats->print_pcie(f1);

      fclose(f1);

      FILE* f2 = fopen("Access_pattern_detail.txt", "w");

      g_the_gpu->m_new_stats->print_access_pattern_detail(f2);

      fclose(f2);
*/

  // Figure 1,6
  FILE* f1 = fopen("fig-access_count.txt", "w");
  new_stats->print_access_pattern(f1);
  fclose(f1);

  // Figure 2
  FILE *f2 = fopen("fig-access.txt", "w");
  new_stats->print_time_and_access(f2);
  fclose(f2);

  // Figure 3,5
  FILE *f3 = fopen("fig-cycle_pf_num.txt","w");
  new_stats->print_cycle_pf_nums(f3);
  fclose(f3);

  FILE *f4 = fopen("fig-cycle_valid_pages.txt","w");
  new_stats->print_cycle_pageset(f4);
  fclose(f4);

  //Figure 3,5
  FILE *f5 = fopen("fig-cycle_pe_num.txt","w");
  new_stats->print_cycle_pe_nums(f5);
  fclose(f5);

  // Figure 7
  FILE *f7 = fopen("fig-page_access_stats.txt","w");
  new_stats->print_page_access_stats(f7);
  fclose(f7);

  FILE *f9 = fopen("fig-page_rw.txt","w");
  new_stats->print_page_read_write(f9);
  fclose(f9);

  if (sim_prof_enable) {
    print_sim_prof(stdout, gpu->shader_clock());
    calculate_sim_prof(stdout, gpu);
  }
}

void calculate_sim_prof(FILE *fout, gpgpu_sim *gpu) {
  float freq = gpu->shader_clock() / 1000.0;
  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      (*iter2)->calculate();
    }
  }

  unsigned long long page_fault_time = 0;
  if (!gpu->get_config().enable_accurate_simulation) {
    page_fault_time = gpu->m_new_stats->mf_page_fault_outstanding *
                      gpu->get_config().page_fault_latency;
  }

  fprintf(fout, "Tot_prefetch_time: %llu(cycle), %f(us)\n", prefetch_time,
          ((float)prefetch_time) / freq);
  fprintf(fout, "Tot_kernel_exec_time: %llu(cycle), %f(us)\n", kernel_time,
          ((float)kernel_time) / freq);

  if (!gpu->get_config().enable_accurate_simulation) {
    fprintf(fout, "Tot_kernel_exec_time_and_fault_time: %llu(cycle), %f(us)\n",
            kernel_time + page_fault_time,
            ((float)(kernel_time + page_fault_time)) / freq);
  }

  fprintf(fout, "Tot_memcpy_h2d_time: %llu(cycle), %f(us)\n",
          memory_copy_time_h2d, ((float)memory_copy_time_h2d) / freq);
  fprintf(fout, "Tot_memcpy_d2h_time: %llu(cycle), %f(us)\n",
          memory_copy_time_d2h, ((float)memory_copy_time_d2h) / freq);
  fprintf(fout, "Tot_memcpy_time: %llu(cycle), %f(us)\n",
          memory_copy_time_h2d + memory_copy_time_d2h,
          ((float)(memory_copy_time_h2d + memory_copy_time_d2h)) / freq);
  fprintf(fout, "Tot_devicesync_time: %llu(cycle), %f(us)\n", devicesync_time,
          ((float)devicesync_time) / freq);
  fprintf(fout, "Tot_writeback_time: %llu(cycle), %f(us)\n", writeback_time,
          ((float)writeback_time) / freq);
  fprintf(fout, "Tot_dma_time: %llu(cycle), %f(us)\n", dma_time,
          ((float)dma_time) / freq);
  fprintf(fout, "Tot_memcpy_d2h_sync_wb_time: %llu(cycle), %f(us)\n",
          writeback_time + devicesync_time + memory_copy_time_d2h,
          ((float)(writeback_time + devicesync_time + memory_copy_time_d2h) /
           freq));
}

void update_sim_prof_kernel(unsigned kernel_id, unsigned long long end_time) {
  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if ((*iter2)->type == kernel_launch &&
          ((kernel_stats *)(*iter2))->kernel_id == kernel_id) {
        (*iter2)->end_time = end_time;
        return;
      }
    }
  }
}

void update_sim_prof_prefetch(mem_addr_t start_addr, size_t size,
                              unsigned long long end_time) {
  for (std::map<unsigned long long, std::list<event_stats *>>::reverse_iterator
           iter = sim_prof.rbegin();
       iter != sim_prof.rend(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if ((*iter2)->type == prefetch &&
          ((memory_stats *)(*iter2))->start_addr == start_addr &&
          ((memory_stats *)(*iter2))->size == size) {
        (*iter2)->end_time = end_time;
        return;
      }
    }
  }
}

void update_sim_prof_prefetch_break_down(unsigned long long end_time) {
  for (std::map<unsigned long long, std::list<event_stats *>>::reverse_iterator
           iter = sim_prof.rbegin();
       iter != sim_prof.rend(); iter++) {
    for (std::list<event_stats *>::reverse_iterator iter2 =
             iter->second.rbegin();
         iter2 != iter->second.rend(); iter2++) {
      if ((*iter2)->type == prefetch_breakdown) {
        (*iter2)->end_time = end_time;
        return;
      }
    }
  }
}

// uvm: ==============================================
gmmu_t::gmmu_t(class gpgpu_sim *gpu, const gpgpu_sim_config &config,
               class gpgpu_new_stats *new_stats)
    : m_gpu(gpu), m_config(config), m_new_stats(new_stats) {
  m_shader_config = &m_config.m_shader_config;

  // pinned memory
  if (m_config.enable_dma == 0) {
    dma_mode = dma_type::DISABLED;
  } else if (m_config.enable_dma == 1) {
    dma_mode = dma_type::ADAPTIVE;
  } else if (m_config.enable_dma == 2) {
    dma_mode = dma_type::ALWAYS;
  } else if (m_config.enable_dma == 3) {
    dma_mode = dma_type::OVERSUB;
  } else {
    printf("Unknown DMA mode\n");
    exit(1);
  }

  if (m_config.eviction_policy == 0) {
    evict_policy = eviction_policy::LRU;
  } else if (m_config.eviction_policy == 1) {
    evict_policy = eviction_policy::TBN;
  } else if (m_config.eviction_policy == 2) {
    evict_policy = eviction_policy::SEQUENTIAL_LOCAL;
  } else if (m_config.eviction_policy == 3) {
    evict_policy = eviction_policy::RANDOM;
  } else if (m_config.eviction_policy == 4) {
    evict_policy = eviction_policy::LFU;
  } else if (m_config.eviction_policy == 5) {
    evict_policy = eviction_policy::LRU4K;
  } else {
    printf("Unknown eviction policy");
    exit(1);
  }

  if (m_config.hardware_prefetch == 0) {
    prefetcher = hwardware_prefetcher::DISBALED;
  } else if (m_config.hardware_prefetch == 1) {
    prefetcher = hwardware_prefetcher::TBN;
  } else if (m_config.hardware_prefetch == 2) {
    prefetcher = hwardware_prefetcher::SEQUENTIAL_LOCAL;
  } else if (m_config.hardware_prefetch == 3) {
    prefetcher = hwardware_prefetcher::RANDOM;
  } else {
    printf("Unknown hardware prefeching policy");
    exit(1);
  }

  if (m_config.hwprefetch_oversub == 0) {
    oversub_prefetcher = hwardware_prefetcher_oversub::DISBALED;
  } else if (m_config.hwprefetch_oversub == 1) {
    oversub_prefetcher = hwardware_prefetcher_oversub::TBN;
  } else if (m_config.hwprefetch_oversub == 2) {
    oversub_prefetcher = hwardware_prefetcher_oversub::SEQUENTIAL_LOCAL;
  } else if (m_config.hwprefetch_oversub == 3) {
    oversub_prefetcher = hwardware_prefetcher_oversub::RANDOM;
  } else {
    printf("Unknown hardware prefeching policy under over-subscription");
    exit(1);
  }

  pcie_read_latency_queue = NULL;
  pcie_write_latency_queue = NULL;

  total_allocation_size = 0;

  over_sub = false;

  //gpu_sim_cycle = m_gpu->gpu_sim_cycle;
  //gpu_tot_sim_cycle = m_gpu->gpu_tot_sim_cycle;
}

unsigned long long gmmu_t::calculate_transfer_time(size_t data_size) {
  float speed = 2.0 * m_config.curve_a / M_PI *
                atan(m_config.curve_b * ((float)(data_size) / (float)(1024)));

  // 2 KB
  if (data_size >= 2 * 1024 * 1024) {
    speed /= 2;
  }

  return (unsigned long long)((float)(data_size)*m_config.core_freq / speed /
                              (1024.0 * 1024.0 * 1024.0));
}

void gmmu_t::calculate_devicesync_time(size_t data_size) {

  unsigned cur_turn = 0;
  unsigned cur_size = 0;

  float speed;

  while (data_size != 0) {

    unsigned long long cur_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
    unsigned long long cur_time = 0;

    if (cur_turn == 0) {
      cur_size = MIN_PREFETCH_SIZE;
    } else {
      cur_size = MIN_PREFETCH_SIZE * pow(2, cur_turn - 1);
    }

    if (data_size < 4096) {
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(data_size) / (float)(1024)));
      cur_time = (unsigned long long)((float)(data_size)*m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, data_size, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;

      return;
    } else {
      cur_size -= 4096;
      data_size -= 4096;
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(4096) / (float)(1024)));
      cur_time = (unsigned long long)((float)(4096) * m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, 4096, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;
    }

    if (data_size < cur_size) {
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(data_size) / (float)(1024)));
      cur_time = (unsigned long long)((float)(data_size)*m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, data_size, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;

      return;
    } else {
      data_size -= cur_size;
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(cur_size) / (float)(1024)));
      cur_time = (unsigned long long)((float)(cur_size)*m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, cur_size, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;
    }

    cur_turn++;
    if (cur_turn == 6) {
      cur_turn = 0;
    }
  }
  return;
}

bool gmmu_t::pcie_transfers_completed() {
  return pcie_write_stage_queue.empty() && pcie_write_latency_queue == NULL &&
         pcie_read_stage_queue.empty() && pcie_read_latency_queue == NULL;
}

void gmmu_t::register_tlbflush_callback(
    std::function<void(mem_addr_t)> cb_tlb) {
  callback_tlb_flush.push_back(cb_tlb);
}

void gmmu_t::tlb_flush(mem_addr_t page_num) {
  for (std::list<std::function<void(mem_addr_t)>>::iterator iter =
           callback_tlb_flush.begin();
       iter != callback_tlb_flush.end(); iter++) {
    (*iter)(page_num);
  }
}

void gmmu_t::check_write_stage_queue(mem_addr_t page_num, bool refresh) {
  // the page, about to be accessed, was selected for eviction earlier
  // so don't evict that page
  for (std::list<pcie_latency_t *>::iterator iter = pcie_write_stage_queue.begin(); iter != pcie_write_stage_queue.end(); iter++) {

    // current PFN in a write page_list
    if (std::find((*iter)->page_list.begin(), (*iter)->page_list.end(), page_num) != (*iter)->page_list.end()) {

      // on tlb hit refresh position of pages in the valid page list
      for (std::list<mem_addr_t>::iterator pg_iter = (*iter)->page_list.begin(); pg_iter != (*iter)->page_list.end(); pg_iter++) {
        
        m_gpu->get_global_memory()->set_page_access(*pg_iter);
        m_gpu->get_global_memory()->set_page_dirty(*pg_iter);

        // reclaim valid size in large page tree for unique basic blocks
        // corresponding to all pages
        mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(*pg_iter);
        struct lp_tree_node *root = get_lp_node(page_addr);
        update_basic_block(root, page_addr, m_config.page_size, true);

        refresh_valid_pages(page_addr);
      }
      // remove it from pcie_ws_queue
      pcie_write_stage_queue.erase(iter);
      break;
    }
  }
}

// check if the block is already scheduled for eviction or is not valid at all
bool gmmu_t::is_block_evictable(mem_addr_t addr, size_t size) {
  // look inside this *block*
  for (mem_addr_t start = addr; start != addr + size; start += m_config.page_size) {
    // check the pfn in the page table is valid
    if (!m_gpu->get_global_memory()->is_valid(m_gpu->get_global_memory()->get_page_num(start))) {
      return false;
    }
  }


  for (std::list<pcie_latency_t *>::iterator iter = pcie_write_stage_queue.begin(); iter != pcie_write_stage_queue.end(); iter++) {
    // in the range of a pcie write queue
    if ((addr >= (*iter)->start_addr) && (addr < (*iter)->start_addr + (*iter)->size)) {
      return false;
    }
  }


  for (mem_addr_t start = addr; start != addr + size; start += m_config.page_size) {
    // whether reserved pages
    if (!reserve_pages_check(start)) {
      return false;
    }
  }

  return true;
}

// 1. get valid pages
// 2. decide evicted pages
// 3. send to pcie write queue
void gmmu_t::page_eviction_procedure() {
  // ----------Step - 1: get the page candidate list for eviction----------
  // not for random, as no need for sort
  // otherwise based on access counters
  // sort valid pages in list
  sort_valid_pages();

  // first : mem_addr
  // second : size, could be one page or a block size
  std::list<std::pair<mem_addr_t, size_t>> evicted_pages;

  // generally from zero
  int eviction_start = (int)(valid_pages.size() * m_config.reserve_accessed_page_percent / 100);


  // just one 4K page
  if (evict_policy == eviction_policy::LRU4K) {

    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    // go eviction_start to the iterator
    std::advance(iter, eviction_start);

    // find a page that is:
    // valid
    // for in pcie write
    // not reserved
    while (iter != valid_pages.end() && !is_block_evictable((*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;

      // this page in the tree
      // get the block (64KB?) contain this page
      struct lp_tree_node *root = get_lp_node(page_addr);
      
      update_basic_block(root, page_addr, m_config.page_size, false);
      // pages to be evicted
      // add modification here if needed
      // e.g. (how many sensitive information inside)
      // one page
      evicted_pages.push_back(std::make_pair(page_addr, m_config.page_size));
    }
  } else if (evict_policy == eviction_policy::LRU ||
             evict_policy == eviction_policy::LFU ||
             m_config.page_size == MAX_PREFETCH_SIZE) {
    // either 64KB ?

    // in lru, only evict the least recently used pages at the front of accessed
    // pages queue in lfu, only evict the page accessed least number of times
    // from the front of accessed pages queue
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    while (iter != valid_pages.end() &&
           !is_block_evictable((*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      evict_whole_tree(root);

      evicted_pages.push_back(std::make_pair(root->addr, root->size));
    }
  } else if (evict_policy == eviction_policy::RANDOM) {
    // in random eviction, select a random page
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    // seems not that random
    // TODO: add real random in it
    std::advance(
        iter, eviction_start +
                  (rand() %
                   (int)(valid_pages.size() *
                         (1 - m_config.reserve_accessed_page_percent / 100))));

    while (iter != valid_pages.end() && !is_block_evictable((*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      update_basic_block(root, page_addr, m_config.page_size, false);
      // one page with size PGSize
      evicted_pages.push_back(std::make_pair(page_addr, m_config.page_size));
    }
  } else if (evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    // 64 KB
    // we evict sixteen 4KB pages in the 2 MB allocation where this evictable belong to
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    struct lp_tree_node *root;
    mem_addr_t page_addr;
    mem_addr_t bb_addr;

    for (; iter != valid_pages.end(); iter++) {
      page_addr = (*iter)->addr;

      root = get_lp_node(page_addr);

      bb_addr = get_basic_block(root, page_addr);

      if (is_block_evictable(bb_addr, MIN_PREFETCH_SIZE)) {
        update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, false);
        break;
      }
    }

    if (iter != valid_pages.end()) {
      evicted_pages.push_back(std::make_pair(bb_addr, MIN_PREFETCH_SIZE));
    }
  } else if (evict_policy == eviction_policy::TBN) {
    // we evict multiple 64KB pages in the 2 MB allocation where this evictable
    // belong
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    // find all basic blocks which are not staged/scheduled for write back or
    // not invalid or not in ld/st unit
    std::set<mem_addr_t> all_basic_blocks;

    struct lp_tree_node *root;
    mem_addr_t page_addr;
    mem_addr_t bb_addr;

    for (; iter != valid_pages.end(); iter++) {
      page_addr = (*iter)->addr;

      root = get_lp_node(page_addr);

      bb_addr = get_basic_block(root, page_addr);

      if (is_block_evictable(bb_addr, MIN_PREFETCH_SIZE)) {
        update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, false);
        break;
      }
    }

    if (iter != valid_pages.end()) {
      all_basic_blocks.insert(bb_addr);
      traverse_and_remove_lp_tree(root, all_basic_blocks);
    }

    // group all contiguous basic blocks if possible
    std::set<mem_addr_t>::iterator bb = all_basic_blocks.begin();

    while (bb != all_basic_blocks.end()) {
      std::set<mem_addr_t>::iterator next_bb = bb;
      size_t cur_num = 0;

      do {
        next_bb++;
        cur_num++;
      } while (next_bb != all_basic_blocks.end() &&
               ((*next_bb) == ((*bb) + cur_num * MIN_PREFETCH_SIZE)));

      evicted_pages.push_back(
          std::make_pair((*bb), (cur_num * MIN_PREFETCH_SIZE)));

      bb = next_bb;
    }
  }

  // ----------Step - 2: create pcie transaction and define latency type for each page in evicted pages----------
  // always write back the chunk no matter what it has not dirty pages or dirty pages
  // for all eviction pages
  // loop because in TBN could be multiple evicted_pages
  for (std::list<std::pair<mem_addr_t, size_t>>::iterator iter = evicted_pages.begin(); iter != evicted_pages.end(); iter++) {
    // create a transaction
    pcie_latency_t *p_t = new pcie_latency_t();
    
    // the start PFN
    p_t->start_addr = iter->first;
    // the size of the block (page list)
    p_t->size = iter->second;
    latency_type ltype = latency_type::PCIE_WRITE_BACK;
    
    // pages that could be evicted
    for (std::list<eviction_t *>::iterator it = valid_pages.begin(); it != valid_pages.end(); it++) {
      // such evicted pages in in a range of valid pages
      if ((*it)->addr <= iter->first && iter->first < (*it)->addr + (*it)->size) {
        // Dirty
        if ((*it)->RW == 1) {
          ltype = latency_type::INVALIDATE;
          break;
        }
      }
    }
    // which event
    p_t->type = ltype;
    // ----------Step - 3: update valid page list----------
    // 2MB page
    if (m_config.page_size == MAX_PREFETCH_SIZE) {
      mem_addr_t page_num =
          m_gpu->get_global_memory()->get_page_num(iter->first);

      p_t->page_list.push_back(page_num);

      valid_pages_erase(page_num);
    } else {
      // 4KB pages
      // get PFN
      mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(iter->first);
      // the block size / page size -> how many pages in such eviction
      for (int i = 0; i < (int)(iter->second / m_config.page_size); i++) {
        p_t->page_list.push_back(page_num + i);
        // remove it from valid
        valid_pages_erase(page_num + i);
      }
    }
    // ----------Step - 4: send this evicted pages to pcie write queue----------
    pcie_write_stage_queue.push_back(p_t);
  }
}

void gmmu_t::valid_pages_erase(mem_addr_t page_num) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  for (std::list<eviction_t *>::iterator it = valid_pages.begin();
       it != valid_pages.end(); it++) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      valid_pages.erase(it);
      break;
    }
  }
}

void gmmu_t::valid_pages_clear() { valid_pages.clear(); }

void gmmu_t::refresh_valid_pages(mem_addr_t page_addr) {
  bool valid = false;

  for (std::list<eviction_t *>::iterator it = valid_pages.begin(); it != valid_pages.end(); it++) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      (*it)->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
      valid = true;
      break;
    }
  }

  if (!valid) {
    eviction_t *item = new eviction_t();
    item->addr = get_eviction_base_addr(page_addr);
    item->size = get_eviction_granularity(page_addr);
    item->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
    valid_pages.push_back(item);
  }
}

void gmmu_t::sort_valid_pages() {
  // keep a lru list
  // update valid page information
  for (std::list<eviction_t *>::iterator vp_iter = valid_pages.begin();vp_iter != valid_pages.end(); vp_iter++) {
    // for each page update counters
    // mathch - update - then next
    for (std::list<struct lp_tree_node *>::iterator lp_iter = large_page_info.begin();lp_iter != large_page_info.end(); lp_iter++) {
      if ((*vp_iter)->addr == (*lp_iter)->addr) {
        (*vp_iter)->access_counter = (*lp_iter)->access_counter;
        (*vp_iter)->RW = (*lp_iter)->RW;
        break;
      }
    }
  }

  if (evict_policy == eviction_policy::LFU) {
    valid_pages.sort([](const eviction_t *i, const eviction_t *j) {
      return (i->access_counter < j->access_counter) ||
             ((i->access_counter == j->access_counter) && (i->RW < j->RW)) ||
             ((i->access_counter == j->access_counter) && (i->RW == j->RW) &&
              (i->cycle < j->cycle));
    });
  } else {
    if (evict_policy == eviction_policy::TBN || evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
      std::map<mem_addr_t, std::list<eviction_t *>> tempMap;

      for (std::list<eviction_t *>::iterator it = valid_pages.begin();
           it != valid_pages.end(); it++) {
        struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node((*it)->addr);
        tempMap[root->addr].push_back(*it);
      }

      for (std::map<mem_addr_t, std::list<eviction_t *>>::iterator it =
               tempMap.begin();
           it != tempMap.end(); it++) {
        it->second.sort([](const eviction_t *i, const eviction_t *j) {
          return i->cycle > j->cycle;
        });
      }

      std::list< std::pair<mem_addr_t, std::list<eviction_t *> > > tempList;

      for (std::map<mem_addr_t, std::list<eviction_t *>>::iterator it =
               tempMap.begin();
           it != tempMap.end(); it++) {
        tempList.push_back(make_pair(it->first, it->second));
      }

      tempList.sort([](const std::pair<mem_addr_t, std::list<eviction_t *>> i,
                       const std::pair<mem_addr_t, std::list<eviction_t *>> j) {
        return i.second.front()->cycle < j.second.front()->cycle;
      });

      std::list<eviction_t *> new_valid_pages;

      for (std::list<std::pair<mem_addr_t, std::list<eviction_t *>>>::iterator it =
               tempList.begin();
           it != tempList.end(); it++) {
        (*it).second.sort([](const eviction_t *i, const eviction_t *j) {
          return i->cycle < j->cycle;
        });
        new_valid_pages.insert(new_valid_pages.end(), it->second.begin(),
                               it->second.end());
      }

      valid_pages = new_valid_pages;
    } else {
      // lru 4k
      valid_pages.sort([](const eviction_t *i, const eviction_t *j) {
        return i->cycle < j->cycle;
      });
    }
  }
}

int gmmu_t::get_trng(int low, int high){
  int rd = -1;
  #include<time.h>
    srand((unsigned int)time(NULL));
    rd = rand() % high + low;
  return rd;
}

// basically transfer time needed
// speed is the effective PCIe bandwidth
// then based on the number of transfered pages,
// the effective PCIe bandwidth is calculated
// then the transfer time is calculated
unsigned long long gmmu_t::get_ready_cycle(unsigned num_pages) {
  
  // atan in -pi/2 to pi/2
  // 2/pi * atan(x) -> -1 to 1
  // curve_a and curve_b for scaling
  // data in kB
  // speed in GB/s
  //float speed = 2.0 * m_config.curve_a / M_PI *atan(m_config.curve_b *((float)(num_pages * m_config.page_size) / 1024.0));
  unsigned long long need_cycles = 0;
  float speed = 2.0 * m_config.curve_a / M_PI * atan (m_config.curve_b * ((float)(num_pages*m_config.page_size)/1024.0) );
  need_cycles = (unsigned long long)((float)(m_config.page_size * num_pages) * m_config.core_freq / speed /(1024.0 * 1024.0 * 1024.0));
  if(new_debug_execution == 1){
    std::cout << "num_pages: " << num_pages << std::endl;
    std::cout << "bw: " << speed << std::endl;
    std::cout << "need_cycle: "<<need_cycles <<std::endl;
    std::cout << "----------" <<std::endl;
  }
  
  // #data in B, 4096 B
  // then / 1024*3 -> get data in GB
  // then / speed -> get time in s
  // then * f -> get how many cycles
  return m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle + need_cycles;
}


unsigned long long gmmu_t::get_ready_cycle(pcie_latency_t * p_pcie_latency_queue){  
  
  if (m_gpu->contain_crypto()){
    // loop for the sensitivity information
    const std::map<uint64_t, struct allocation_info *> &managedAllocations = m_gpu->gpu_get_managed_allocations();

    // range is start, start+size
    mem_addr_t start_addr = m_gpu->get_global_memory()->get_mem_addr(p_pcie_latency_queue->page_list.front());
    mem_addr_t end_addr = start_addr + m_config.page_size * p_pcie_latency_queue->page_list.size();

    std::cout << "page_start_addr: " << std::showbase << std::hex << start_addr << " page_end_addr: "<< std::showbase << std::hex << end_addr << " real_size: "<< p_pcie_latency_queue->size <<std::endl;
    m_gpu->print_managed_allocations();
    std::vector< std::pair<enum crypto_level, float> > how_many_sens;
    // 128b cipher
    int block_size = 16;
    std::map<enum crypto_level, size_t > crypto_overhead;

    unsigned long long calculated_size = 0;
    float crypto_cal_size = 0.0;
    unsigned long long remained_normal_size = 0;

    for(auto iter = managedAllocations.begin(); iter!= managedAllocations.end();++iter){
        // gpu_mem_addr and alocated size
        struct allocation_info* ali = iter->second;
        struct sensitivity_info si = ali->si;
        
        enum crypto_level cl = si._cl;
        int protect = si._protected?1:0;
        int protect_perc = si.protect_percentage;
        
        uint64_t gpu_start_addr = ali->gpu_mem_addr;
        size_t alloc_size = ali->allocation_size;
        uint64_t gpu_end_addr = gpu_start_addr + alloc_size;

        float crypto_protect_size = 0.0;

        bool region1 = ( start_addr <= gpu_start_addr && gpu_start_addr <=end_addr ) && ( start_addr <= gpu_end_addr && gpu_end_addr <=end_addr);
        bool region2 = ( gpu_start_addr <= start_addr && gpu_start_addr <=end_addr ) && ( start_addr <= gpu_end_addr && gpu_end_addr <=end_addr);
        bool region3 = ( start_addr <= gpu_start_addr && gpu_start_addr <=end_addr ) && ( start_addr <= gpu_end_addr && end_addr <=gpu_end_addr);
        bool region4 = ( gpu_start_addr <= start_addr && gpu_start_addr <=end_addr ) && ( start_addr <= gpu_end_addr && end_addr <=gpu_end_addr);

        if (region1 || region2 || region3 || region4){
          unsigned long long left = std::max((unsigned long long)start_addr,(unsigned long long)gpu_start_addr);
          unsigned long long right = std::min((unsigned long long)end_addr,(unsigned long long)gpu_end_addr);
          size_t processed_size = right - left;
          crypto_protect_size = (protect_perc/100.0) * processed_size * protect;
          how_many_sens.push_back(std::make_pair(cl,crypto_protect_size));
          // how many 16B blocks -> calculate size
          crypto_overhead[cl]+= std::ceil((crypto_protect_size/block_size))*block_size;
          calculated_size += processed_size;
          crypto_cal_size += crypto_protect_size;
        }
    }
    printf("calculated size %d\n",calculated_size);
    printf("page_list size %d\n",m_config.page_size * p_pcie_latency_queue->page_list.size());
    printf("crypto_protect_size %f\n",crypto_cal_size);
    assert(calculated_size == m_config.page_size * p_pcie_latency_queue->page_list.size());
    // for simple, whole size * protect_percent
    remained_normal_size = calculated_size - std::ceil(crypto_cal_size);
    printf ("remained_normal_size %llu\n",remained_normal_size);
    assert(remained_normal_size>0);

    unsigned tot_cycles = 0;

    // remainning size in KB
    float speed = 2.0 * m_config.curve_a / M_PI * atan (m_config.curve_b * ((float)(remained_normal_size)/1024.0) );
    unsigned long long unprotected_cycles =  (unsigned long long)((float)(remained_normal_size) * m_config.core_freq / speed /(1024.0 * 1024.0 * 1024.0));
    tot_cycles+= unprotected_cycles;
    // crypto cycles
    for (auto cit = crypto_overhead.begin();cit!=crypto_overhead.end();cit++){
      enum crypto_level cl = cit->first;
      float bw = crypto_bw[cl];
      size_t size = cit->second;
      unsigned long long protected_cycles = (unsigned long long)((float)(size) * m_config.core_freq / bw /(1024.0 * 1024.0 * 1024.0));
      printf ("crypto level %s, overhead %llu\n",crypto_level_string[cl],protected_cycles);
      tot_cycles += protected_cycles;
    }

    return m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle + tot_cycles;
  }else{
    return get_ready_cycle(p_pcie_latency_queue->page_list.size());
  }
}



// just 200 cycles for DMA
unsigned long long gmmu_t::get_ready_cycle_dma(unsigned size) {
  float speed = 2.0 * m_config.curve_a / M_PI *
                atan(m_config.curve_b * ((float)(size) / 1024.0));
  return m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle + 200;
}

// Bandwidth utilization calculation

float gmmu_t::get_pcie_utilization(unsigned num_pages) {
  //
  return 2.0 * m_config.curve_a / M_PI * atan(m_config.curve_b *((float)(num_pages * m_config.page_size) / 1024.0)) / m_config.pcie_bandwidth;
}

void gmmu_t::activate_prefetch(mem_addr_t m_device_addr, size_t m_cnt,
                               struct CUstream_st *m_stream) {
  for (std::list<prefetch_req>::iterator iter = prefetch_req_buffer.begin(); iter != prefetch_req_buffer.end(); iter++) {
    if (iter->start_addr == m_device_addr && iter->size == m_cnt && iter->m_stream->get_uid() == m_stream->get_uid()) {
      assert(iter->cur_addr == m_device_addr);
      iter->active = true;
      return;
    }
  }
}

void gmmu_t::register_prefetch(mem_addr_t m_device_addr,
                               mem_addr_t m_device_allocation_ptr, size_t m_cnt,
                               struct CUstream_st *m_stream) {
  struct prefetch_req pre_q;

  pre_q.start_addr = m_device_addr;
  pre_q.cur_addr = m_device_addr;
  pre_q.allocation_addr = m_device_allocation_ptr;
  pre_q.size = m_cnt;
  pre_q.active = false;
  pre_q.m_stream = m_stream;

  prefetch_req_buffer.push_back(pre_q);
}

struct lp_tree_node *gmmu_t::build_lp_tree(mem_addr_t addr, size_t size) {
  struct lp_tree_node *node = new lp_tree_node();
  node->addr = addr;
  node->size = size;
  node->valid_size = 0;
  node->access_counter = 0;
  node->RW = 0;

  if (size == MIN_PREFETCH_SIZE) {
    node->left = NULL;
    node->right = NULL;
  } else {
    node->left = build_lp_tree(addr, size / 2);
    node->right = build_lp_tree(addr + size / 2, size / 2);
  }
  return node;
}

void gmmu_t::initialize_large_page(mem_addr_t start_addr, size_t size) {
  struct lp_tree_node *root = build_lp_tree(start_addr, size);

  large_page_info.push_back(root);

  total_allocation_size += size;
}

struct lp_tree_node *gmmu_t::get_lp_node(mem_addr_t addr) {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    if ((*iter)->addr <= addr && addr < (*iter)->addr + (*iter)->size) {
      return *iter;
    }
  }
  return NULL;
}

mem_addr_t gmmu_t::get_basic_block(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return node->addr;
}

void gmmu_t::evict_whole_tree(struct lp_tree_node *node) {
  if (node != NULL) {
    node->valid_size = 0;
    evict_whole_tree(node->left);
    evict_whole_tree(node->right);
  }
}

mem_addr_t gmmu_t::update_basic_block(struct lp_tree_node *node,
                                      mem_addr_t addr, 
                                      size_t size,
                                      bool prefetch) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (prefetch) {
      if (node->valid_size != node->size) {
        node->valid_size += size;
      }
    } else {
      // reduce valid size after evict a *block*
      if (node->valid_size != 0) {
        node->valid_size -= size;
      }
    }

    // update nodes [every level then reduce valid size]
    if (node->left->addr <= addr && addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  // the leaf is 64KB block
  if (prefetch) {
    if (node->valid_size != node->size) {
      node->valid_size += size;
    }
  } else {
    if (node->valid_size != 0) {
      node->valid_size -= size;
    }
  }

  return node->addr;
}

void gmmu_t::fill_lp_tree(struct lp_tree_node *node,
                          std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size == MIN_PREFETCH_SIZE) {
    if (node->valid_size == 0) {
      node->valid_size = MIN_PREFETCH_SIZE;
      scheduled_basic_blocks.insert(node->addr);
    }
  } else {
    fill_lp_tree(node->left, scheduled_basic_blocks);
    fill_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
  }
}

void gmmu_t::remove_lp_tree(struct lp_tree_node *node,
                            std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size == MIN_PREFETCH_SIZE) {
    if (node->valid_size == MIN_PREFETCH_SIZE &&
        is_block_evictable(node->addr, MIN_PREFETCH_SIZE)) {
      node->valid_size = 0;
      scheduled_basic_blocks.insert(node->addr);
    }
  } else {
    remove_lp_tree(node->left, scheduled_basic_blocks);
    remove_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
  }
}

void gmmu_t::traverse_and_fill_lp_tree(
    struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size != MIN_PREFETCH_SIZE) {
    traverse_and_fill_lp_tree(node->left, scheduled_basic_blocks);
    traverse_and_fill_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
    // seems here is the 50% logic on TBNp threshold
    if (node->valid_size != node->size && node->valid_size > node->size / 2) {
      fill_lp_tree(node, scheduled_basic_blocks);
    }
  }
}

void gmmu_t::traverse_and_remove_lp_tree(
    struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size != MIN_PREFETCH_SIZE) {
    traverse_and_remove_lp_tree(node->left, scheduled_basic_blocks);
    traverse_and_remove_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;

    if (node->valid_size != 0 && node->valid_size < node->size / 2) {
      remove_lp_tree(node, scheduled_basic_blocks);
    }
  }
}

void gmmu_t::reserve_pages_insert(mem_addr_t addr, unsigned ma_uid) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);

  if (find(reserve_pages[page_num].begin(), reserve_pages[page_num].end(),ma_uid) == reserve_pages[page_num].end()) {
    reserve_pages[page_num].push_back(ma_uid);
    if (g_debug_execution >= 5){
      std::cout << "insert a page to reserve_pages[" << page_num <<"]" <<"with ma_uid "<<ma_uid<< std::endl;
      // print reserve_pages
      for (std::map<mem_addr_t, std::list<unsigned>>::iterator it = reserve_pages.begin(); it != reserve_pages.end(); ++it) {
        std::cout << it->first << " : ";
        for (std::list<unsigned>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
          std::cout << *it2 << " ";
        }
        std::cout << std::endl;
      }
    }
  }
}

void gmmu_t::reserve_pages_remove(mem_addr_t addr, unsigned ma_uid) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);
  fflush(stdout);
  assert(reserve_pages.find(page_num) != reserve_pages.end());
  std::list<unsigned>::iterator iter = std::find(
      reserve_pages[page_num].begin(), reserve_pages[page_num].end(), ma_uid);
  assert(iter != reserve_pages[page_num].end());
  reserve_pages[page_num].erase(iter);
  if (reserve_pages[page_num].empty()) {
    reserve_pages.erase(page_num);
  }
  if (g_debug_execution >= 5){
      std::cout << "remove a page from reserve_pages[" << page_num <<"]" <<"with ma_uid "<<ma_uid<<std::endl;
      for (std::map<mem_addr_t, std::list<unsigned>>::iterator it = reserve_pages.begin(); it != reserve_pages.end(); ++it) {
        std::cout << it->first << " : ";
        for (std::list<unsigned>::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
          std::cout << *it2 << " ";
        }
        std::cout << std::endl;
      }
    }
}

bool gmmu_t::reserve_pages_check(mem_addr_t addr) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);

  return reserve_pages.find(page_num) == reserve_pages.end();
}

void gmmu_t::update_hardware_prefetcher_oversubscribed() {
  if (oversub_prefetcher == hwardware_prefetcher_oversub::DISBALED) {
    prefetcher = hwardware_prefetcher::DISBALED;
  } else if (oversub_prefetcher == hwardware_prefetcher_oversub::TBN) {
    prefetcher = hwardware_prefetcher::TBN;
  } else if (oversub_prefetcher ==
             hwardware_prefetcher_oversub::SEQUENTIAL_LOCAL) {
    prefetcher = hwardware_prefetcher::SEQUENTIAL_LOCAL;
  } else if (oversub_prefetcher == hwardware_prefetcher_oversub::RANDOM) {
    prefetcher = hwardware_prefetcher::RANDOM;
  }
}

void gmmu_t::log_kernel_info(unsigned kernel_id, unsigned long long time,
                             bool finish) {
  if (!finish) {
    kernel_info.insert(std::make_pair(kernel_id, std::make_pair(time, 0)));
  } else {
    std::map<unsigned,
             std::pair<unsigned long long, unsigned long long>>::iterator it =
        kernel_info.find(kernel_id);
    if (it != kernel_info.end()) {
      it->second.second = time;
    }
  }
}

// IPDPS'20 paper
void gmmu_t::update_memory_management_policy() {

  std::map<std::string, ds_pattern> accessPatterns;

  int i = 1;
  
  std::map< std::pair<mem_addr_t, size_t>, std::string> dataStructures;
  std::map<std::string, std::list<mem_addr_t>> dsUniqueBlocks;

  // get the managed allocations
  const std::map<uint64_t, struct allocation_info *> &managedAllocations = m_gpu->gpu_get_managed_allocations();

  // loop over managed allocations to create three maps
  // 1. data structures - key: pair of start addr and size; value: ds_i
  // 2. access pattern: key: ds_i; value: UNDECIDED pattern
  // 3. unique accessed blocks for reuse: key: ds_i; value: empty list of block start address
  for (std::map<uint64_t, struct allocation_info *>::const_iterator iter = managedAllocations.begin(); iter != managedAllocations.end(); iter++) {
    dataStructures.insert(
        std::make_pair(std::make_pair(iter->second->gpu_mem_addr,
                                      iter->second->allocation_size),
                       std::string("ds" + std::to_string(i))));

    accessPatterns.insert(std::make_pair(std::string("ds" + std::to_string(i)),
                                         ds_pattern::UNDECIDED));
    dsUniqueBlocks.insert(std::make_pair(std::string("ds" + std::to_string(i)),
                                         std::list<mem_addr_t>()));
    i++;
  }

  // create three level hierarchy for kernel-wise then data-structure wise block
  // address first level: name of kernel (k_i); second level: ds_i; third level:
  // block addresses ordered by access time
  std::map<unsigned, std::map<std::string, std::list<mem_addr_t>>>
      kernel_pattern;

  for (std::map<unsigned,
                std::pair<unsigned long long, unsigned long long>>::iterator
           k_iter = kernel_info.begin();
       k_iter != kernel_info.end(); k_iter++) {

    unsigned long long start = k_iter->second.first;
    unsigned long long end = k_iter->second.second;

    std::map<std::string, std::list<mem_addr_t>> dsAccess;

    for (std::list<std::pair<unsigned long long, mem_addr_t>>::iterator
             acc_iter = block_access_list.begin();
         acc_iter != block_access_list.end(); acc_iter++) {

      unsigned long long access_cycle = acc_iter->first;
      mem_addr_t block_addr = acc_iter->second;

      if (access_cycle >= start && ((end == 0) || (access_cycle <= end))) {

        for (std::map<std::pair<mem_addr_t, size_t>, std::string>::iterator
                 ds_iter = dataStructures.begin();
             ds_iter != dataStructures.end(); ds_iter++) {

          if (block_addr >= ds_iter->first.first &&
              block_addr < ds_iter->first.first + ds_iter->first.second) {
            dsAccess[ds_iter->second].push_back(block_addr);
          }
        }
      }
    }

    kernel_pattern.insert(std::make_pair(k_iter->first, dsAccess));
  }

  // determine pattern per data structure
  // first loop on kernel level then on data structures accessed in that kernel
  for (std::map<unsigned,
                std::map<std::string, std::list<mem_addr_t>>>::iterator k_iter =
           kernel_pattern.begin();
       k_iter != kernel_pattern.end(); k_iter++) {

    for (std::map<std::string, std::list<mem_addr_t>>::iterator da_iter =
             k_iter->second.begin();
         da_iter != k_iter->second.end(); da_iter++) {

      // get the sorted list of block addresses belonging to the current
      // data-structure in current kernel
      std::list<mem_addr_t> curBlocks = std::list<mem_addr_t>(da_iter->second);
      curBlocks.sort();
      curBlocks.unique();

      // check for data reuse
      bool reuse = false;

      // first within this kernel
      // if the number of unique blocks accessed and total number of blocks
      // accessed are not same then there is repetition
      if (curBlocks.size() != da_iter->second.size()) {
        reuse = true;
      }

      // second check if the current accessed blocks are already seen or not
      std::map<std::string, std::list<mem_addr_t>>::iterator ub_it =
          dsUniqueBlocks.find(da_iter->first);

      // check for intersection between unique blocks accessed in current kernel
      // and the previous kernels is null set or not
      std::list<int> intersection;
      std::set_intersection(curBlocks.begin(), curBlocks.end(),
                            ub_it->second.begin(), ub_it->second.end(),
                            std::back_inserter(intersection));

      if (intersection.size() != 0) {
        reuse = true;
      }

      // add the current blocks to the seen set per data structure
      ub_it->second.merge(curBlocks);
      ub_it->second.sort();
      ub_it->second.unique();

      // now update the pattern
      std::map<std::string, ds_pattern>::iterator dsp_it =
          accessPatterns.find(da_iter->first);
      ds_pattern curPattern;

      // check for linearity or randomness in current kernel
      if (std::is_sorted(da_iter->second.begin(), da_iter->second.end())) {
        if (reuse) {
          curPattern = ds_pattern::LINEAR_REUSE;
        } else {
          curPattern = ds_pattern::LINEAR;
        }
      } else {
        if (reuse) {
          curPattern = ds_pattern::RANDOM_REUSE;
        } else {
          curPattern = ds_pattern::RANDOM;
        }
      }

      // determine the pattern
      if (dsp_it->second == ds_pattern::UNDECIDED) {
        dsp_it->second = curPattern;
      } else if (dsp_it->second == ds_pattern::LINEAR) {
        if (curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::LINEAR_REUSE;
        } else if (curPattern == ds_pattern::RANDOM) {
          dsp_it->second = ds_pattern::MIXED;
        } else if (curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::LINEAR_REUSE) {
        if (curPattern == ds_pattern::RANDOM ||
            curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::RANDOM) {
        if (curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::RANDOM_REUSE;
        } else if (curPattern == ds_pattern::LINEAR) {
          dsp_it->second = ds_pattern::MIXED;
        } else if (curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::RANDOM_REUSE) {
        if (curPattern == ds_pattern::LINEAR ||
            curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      }
    }
  }

  bool is_random = false, is_random_reuse = false, is_linear = false,
       is_linear_reuse = false, is_mixed = false, is_mixed_reuse = false;

  for (std::map<std::string, ds_pattern>::iterator ap_iter =
           accessPatterns.begin();
       ap_iter != accessPatterns.end(); ap_iter++) {
    if (ap_iter->second == ds_pattern::RANDOM) {
      is_random = true;
    } else if (ap_iter->second == ds_pattern::RANDOM_REUSE) {
      is_random_reuse = true;
    } else if (ap_iter->second == ds_pattern::LINEAR) {
      is_linear = true;
    } else if (ap_iter->second == ds_pattern::LINEAR_REUSE) {
      is_linear_reuse = true;
    } else if (ap_iter->second == ds_pattern::MIXED) {
      is_mixed = true;
    } else if (ap_iter->second == ds_pattern::MIXED_REUSE) {
      is_mixed_reuse = true;
    }
  }

  if (is_random || is_random_reuse || is_mixed || is_mixed_reuse) {
    dma_mode = dma_type::OVERSUB;
    evict_policy = eviction_policy::TBN;
  } else if (is_linear_reuse) {
    evict_policy = eviction_policy::TBN;
  }
}

void gmmu_t::reset_lp_tree_node(struct lp_tree_node *node) {
  node->valid_size = 0;
  node->access_counter = 0;
  node->RW = 0;

  if (node->size != MIN_PREFETCH_SIZE) {
    reset_lp_tree_node(node->left);
    reset_lp_tree_node(node->right);
  }
}

void gmmu_t::reset_large_page_info() {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    reset_lp_tree_node(*iter);
  }

  over_sub = false;
}

mem_addr_t gmmu_t::get_eviction_base_addr(mem_addr_t page_addr) {
  mem_addr_t lru_addr;

  struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(page_addr);

  if (evict_policy == eviction_policy::TBN ||
      evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    lru_addr = m_gpu->getGmmu()->get_basic_block(root, page_addr);
  } else if (evict_policy == eviction_policy::LRU4K ||
             evict_policy == eviction_policy::RANDOM) {
    lru_addr = page_addr;
  } else {
    lru_addr = root->addr;
  }

  return lru_addr;
}

size_t gmmu_t::get_eviction_granularity(mem_addr_t page_addr) {
  size_t lru_size;

  struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(page_addr);

  if (evict_policy == eviction_policy::TBN ||
      evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    lru_size = MIN_PREFETCH_SIZE;
  } else if (evict_policy == eviction_policy::LRU4K ||
             evict_policy == eviction_policy::RANDOM) {
    lru_size = m_config.page_size;
  } else {
    lru_size = root->size;
  }

  return lru_size;
}

void gmmu_t::update_access_type(mem_addr_t addr, int type) {
  struct lp_tree_node *node = m_gpu->getGmmu()->get_lp_node(addr);

  while (node->size != MIN_PREFETCH_SIZE) {
    node->RW |= type;

    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  node->RW |= type;
}

int gmmu_t::get_bb_access_counter(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return node->access_counter & ((1 << 27) - 1);
}

int gmmu_t::get_bb_round_trip(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
}

void gmmu_t::inc_bb_access_counter(mem_addr_t addr) {
  struct lp_tree_node *node = m_gpu->getGmmu()->get_lp_node(addr);

  while (node->size != MIN_PREFETCH_SIZE) {
    node->access_counter++;

    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  if (node->access_counter == ((1 << 27) - 1)) {
    reset_bb_access_counter();
  }

  node->access_counter++;
}

void gmmu_t::inc_bb_round_trip(struct lp_tree_node *node) {
  if (node->size != MIN_PREFETCH_SIZE) {
    inc_bb_round_trip(node->left);
    inc_bb_round_trip(node->right);
  } else {
    uint16_t round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;

    if (round_trip == ((1 << 6) - 1)) {
      reset_bb_round_trip();
    }

    round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
    round_trip++;

    node->access_counter =
        (round_trip << 27) | (node->access_counter & ((1 << 27) - 1));
  }
}

void gmmu_t::traverse_and_reset_access_counter(struct lp_tree_node *node) {
  if (node->size == MIN_PREFETCH_SIZE) {
    int round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
    int access_counter = (node->access_counter & ((1 << 27) - 1)) >> 1;

    node->access_counter = (round_trip << 27) | access_counter;
  } else {
    traverse_and_reset_access_counter(node->left);
    traverse_and_reset_access_counter(node->right);
    node->access_counter = node->access_counter >> 1;
  }
}

void gmmu_t::reset_bb_access_counter() {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    traverse_and_reset_access_counter(*iter);
  }
}

void gmmu_t::traverse_and_reset_round_trip(struct lp_tree_node *node) {
  if (node->size == MIN_PREFETCH_SIZE) {
    int round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 28;
    int access_counter = node->access_counter & ((1 << 27) - 1);

    node->access_counter = (round_trip << 27) | access_counter;
  } else {
    traverse_and_reset_access_counter(node->left);
    traverse_and_reset_access_counter(node->right);
  }
}

void gmmu_t::reset_bb_round_trip() {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    traverse_and_reset_round_trip(*iter);
  }
}

bool gmmu_t::should_cause_page_migration(mem_addr_t addr, bool is_write) {
  if (dma_mode == dma_type::DISABLED) {
    return true;
  } else if (dma_mode == dma_type::ALWAYS) {
    if (is_write) {
      return true;
    } else {
      struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(addr);

      if (get_bb_access_counter(root, addr) < m_config.migrate_threshold) {
        return false;
      } else {
        return true;
      }
    }
  } else if (dma_mode == dma_type::OVERSUB) {
    if (over_sub) {
      if (is_write) {
        return true;
      } else {
        struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(addr);

        if (get_bb_access_counter(root, addr) < m_config.migrate_threshold) {
          return false;
        } else {
          return true;
        }
      }
    } else {
      return true;
    }
  } else if (dma_mode == dma_type::ADAPTIVE) {
    if (is_write) {
      return true;
    } else {
      struct lp_tree_node *root = m_gpu->getGmmu()->get_lp_node(addr);

      int derived_threshold;

      if (over_sub) {
        derived_threshold = m_config.migrate_threshold *
                            m_config.multiply_dma_penalty *
                            (get_bb_round_trip(root, addr) + 1);
      } else {
        size_t num_read_stage_queue = 0;

        for (std::list<pcie_latency_t *>::iterator iter =
                 pcie_read_stage_queue.begin();
             iter != pcie_read_stage_queue.end(); iter++) {
          num_read_stage_queue += (*iter)->page_list.size();
        }

        size_t num_write_stage_queue = 0;

        for (std::list<pcie_latency_t *>::iterator iter =
                 pcie_write_stage_queue.begin();
             iter != pcie_write_stage_queue.end(); iter++) {
          num_write_stage_queue += (*iter)->page_list.size();
        }

        derived_threshold =
            (int)(1.0 + m_config.migrate_threshold *
                            m_gpu->get_global_memory()->get_projected_occupancy(
                                num_read_stage_queue, num_write_stage_queue,
                                m_config.free_page_buffer_percentage));
      }

      if (get_bb_access_counter(root, addr) < derived_threshold) {
        return false;
      } else {
        return true;
      }
    }
  }
}
// -------------------------------------------------------------------------
// all gmmu operations
// at the beginning of the gpu cycles
void gmmu_t::cycle() {
  int simt_cluster_id = 0;

  size_t num_read_stage_queue = 0;

  for (std::list<pcie_latency_t *>::iterator iter = pcie_read_stage_queue.begin(); iter != pcie_read_stage_queue.end(); iter++) {
    num_read_stage_queue += (*iter)->page_list.size();
  }

  // how many pages
  size_t num_write_stage_queue = 0;

  for (std::list<pcie_latency_t *>::iterator iter = pcie_write_stage_queue.begin(); iter != pcie_write_stage_queue.end(); iter++) {
    num_write_stage_queue += (*iter)->page_list.size();
  }

  num_write_stage_queue += pcie_write_latency_queue != NULL ? pcie_write_latency_queue->page_list.size(): 0;


  // per 1000 cycles
  // record valid pages in current global space
  m_gpu->get_global_memory()->record_resides_pages(m_new_stats->cycle_pages_valid[(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle)/1000]);

  // ----------------------------------------
  // under-oversubscription and memory contention
  if (m_gpu->get_global_memory()->should_evict_page(num_read_stage_queue, num_write_stage_queue, m_config.free_page_buffer_percentage)) {
    
    // memory access pattern detection
    if (m_config.enable_smart_runtime) {
      update_memory_management_policy();
    }
    // seems to be oversub
    // push to pcie_write_stage_queue
    page_eviction_procedure();
  }
  // ----------------------------------------
  // check whether current transfer in the pcie write latency queue is finished
  // reach the finish cycle
  // then where is this ready cycle set? 

  // e.g. wait the eviction done
  if (pcie_write_latency_queue != NULL &&
    (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >= pcie_write_latency_queue->ready_cycle) {
    
    // process all in the queue
    for (std::list<mem_addr_t>::iterator iter = pcie_write_latency_queue->page_list.begin(); iter != pcie_write_latency_queue->page_list.end(); iter++) {
      // perform write back
      // func sim
      m_gpu->gpu_writeback(m_gpu->get_global_memory()->get_mem_addr(*iter));
    }
    
    if (sim_prof_enable) {
      for (std::list<event_stats *>::iterator iter = writeback_stats.begin();iter != writeback_stats.end(); iter++) {
        if (((memory_stats *)(*iter))->start_addr ==
            m_gpu->get_global_memory()->get_mem_addr(
                pcie_write_latency_queue->page_list.front())) {
          event_stats *wb = *iter;
          wb->end_time = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
          sim_prof[wb->start_time].push_back(wb);
          writeback_stats.erase(iter);
          break;
        }
      }
    }

    pcie_write_latency_queue = NULL;
  }
  // ----------------------------------------
  // schedule a write back transfer if there is a write back request in staging [e.g. from eviction]
  // queue and a free lane
  // pcie_write_latency_queue see last step
  if (!pcie_write_stage_queue.empty() && pcie_write_latency_queue == NULL) {

    // get the element event
    // e.g. a eviction transaction
    pcie_write_latency_queue = pcie_write_stage_queue.front();
    // when it will be ready
    // calculate the cycles
    // then the pcie model should be standalone and adapts to eviction well
    // sensitivity information should be added here
    // here only size is included, we consider more information
    // different PFN (addr) binds to different sensitivity
    // that should comes from a global maps when allocating datas
    // for motivation test
    // we could assume inside each page, how sensitivity changes
    //pcie_write_latency_queue->ready_cycle = get_ready_cycle(pcie_write_latency_queue->page_list.size());
    pcie_write_latency_queue->ready_cycle = get_ready_cycle(pcie_write_latency_queue);

    // until it's ready
    for (unsigned long long write_period = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle; 
                            write_period != pcie_write_latency_queue->ready_cycle; write_period++){
      // calculate pcie util
      // also adapts to the pcie model
      m_new_stats->pcie_write_utilization.push_back(std::make_pair(write_period,get_pcie_utilization(pcie_write_latency_queue->page_list.size())));
    }

    // for all pages related in this latency queue
    for (std::list<mem_addr_t>::iterator iter = pcie_write_latency_queue->page_list.begin(); iter != pcie_write_latency_queue->page_list.end(); iter++) {
      // not thrashing
      // true in read stage
      m_new_stats->page_thrashing[*iter].push_back(false);

      // wb for dirty pages or not
      if (m_gpu->get_global_memory()->is_page_dirty(*iter)) {
        m_new_stats->page_evict_dirty++;
      } else {
        m_new_stats->page_evict_not_dirty++;
      }

      m_new_stats->cycle_pe_times[(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle)/1000].push_back(*iter);

      // invalidate the page in page table [m_data]
      m_gpu->get_global_memory()->invalidate_page(*iter);
      m_gpu->get_global_memory()->clear_page_dirty(*iter);
      m_gpu->get_global_memory()->clear_page_access(*iter);

      // release and get a free page
      m_gpu->get_global_memory()->free_pages(1);
      
      // address translation disabled
      // next time access cause ptw
      tlb_flush(*iter);
    }

    // for a set of pages
    struct lp_tree_node *root =
        m_gpu->getGmmu()->get_lp_node(m_gpu->get_global_memory()->get_mem_addr(
            pcie_write_latency_queue->page_list.front()));
    
    // IPDPS'20 paper
    // used for migration counters
    inc_bb_round_trip(root);

    if (sim_prof_enable) {
      if (pcie_write_latency_queue->type == latency_type::INVALIDATE &&
          m_config.invalidate_clean) {
        event_stats *inv = new memory_stats(
            invalidate, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            m_gpu->get_global_memory()->get_mem_addr(
                pcie_write_latency_queue->page_list.front()),
            pcie_write_latency_queue->page_list.size() * m_config.page_size, 0);
        sim_prof[inv->start_time].push_back(inv);
      } else {
        event_stats *wb = new memory_stats(
            write_back, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            m_gpu->get_global_memory()->get_mem_addr(
                pcie_write_latency_queue->page_list.front()),
            pcie_write_latency_queue->page_list.size() * m_config.page_size, 0);
        writeback_stats.push_back(wb);
      }
    }

    // deal with a transaction
    pcie_write_stage_queue.pop_front();

    // dirty write is mandantory see [IPDPS'20]
    // set to NULL
    // then don't wait the ready cycles
    if (pcie_write_latency_queue->type == latency_type::INVALIDATE && m_config.invalidate_clean) {
      pcie_write_latency_queue = NULL;
    }
  }
  // ----------------------------------------
  std::list<mem_addr_t> page_finsihed_for_mf;
  // check whether the current transfer in the pcie latency queue is finished
  // wait until the ready cycle
  if (pcie_read_latency_queue != NULL && 
      (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >= pcie_read_latency_queue->ready_cycle) {
    

    if (pcie_read_latency_queue->type == latency_type::PCIE_READ) {

      for (std::list<mem_addr_t>::iterator iter = pcie_read_latency_queue->page_list.begin();
                                          iter != pcie_read_latency_queue->page_list.end(); iter++) {
        
        // validate the page in page table
        m_gpu->get_global_memory()->validate_page(*iter);

        // add to the valid pages list
        refresh_valid_pages(m_gpu->get_global_memory()->get_mem_addr(*iter));

        m_new_stats->page_thrashing[*iter].push_back(true);

        // check if the transferred page is part of a prefetch request
        if (!prefetch_req_buffer.empty()) {

          prefetch_req &pre_q = prefetch_req_buffer.front();

          // whether in prefetch request
          std::list<mem_addr_t>::iterator iter2 = find(pre_q.pending_prefetch.begin(), pre_q.pending_prefetch.end(),*iter);

          // find it
          if (iter2 != pre_q.pending_prefetch.end()) {

            // pending prefetch holds the list of 4KB pages of a big chunk of
            // tranfer (max upto 2MB) remove it from the list as the PCI-e has
            // transferred the page
            // remove one finished page from the pending prefetch list
            pre_q.pending_prefetch.erase(iter2);

            // if this page is part of current prefecth request
            // add all the dependant memory requests to the
            // outgoing_replayable_nacks these should be replayed only when
            // current block of memory transfer is finished
            pre_q.outgoing_replayable_nacks[*iter].merge(req_info[*iter]);

            // erase the page from the MSHR map
            req_info.erase(req_info.find(*iter));

            skip_cycles = false;

            m_new_stats->pf_page_fault_latency[*iter].back() =
                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
                m_new_stats->pf_page_fault_latency[*iter].back();
          }
        }

        // this page request is created by core on page fault and not part of a
        // prefetch
        if (req_info.find(*iter) != req_info.end()) {

          page_finsihed_for_mf.push_back(*iter);

          // for all memory fetches that were waiting for this page, should be
          // replayed back for cache access
          for (std::list<mem_fetch *>::iterator iter2 = req_info[*iter].begin();
               iter2 != req_info[*iter].end(); iter2++) {
            mem_fetch *mf = *iter2;

            simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();

            // push the memory fetch into the gmmu to cu queue
            (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);
          }

          // erase the page from the MSHR map
          req_info.erase(req_info.find(*iter));

          skip_cycles = false;

          m_new_stats->mf_page_fault_latency[*iter].back() =
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
              m_new_stats->pf_page_fault_latency[*iter].back();
        }
      }
    } else if (pcie_read_latency_queue->type == latency_type::PAGE_FAULT) { 
      // processed far-fault is returned to upward queue
      if (sim_prof_enable) {
        for (std::list<event_stats *>::iterator iter = fault_stats.begin(); iter != fault_stats.end(); iter++) {
          if (((page_fault_stats *)(*iter))->transfering_pages.front() == pcie_read_latency_queue->page_list.front()) {
            event_stats *mf_fault = *iter;
            mf_fault->end_time = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
            sim_prof[mf_fault->start_time].push_back(mf_fault);
            fault_stats.erase(iter);
            break;
          }
        }
      }
    } else if (pcie_read_latency_queue->type == latency_type::DMA) { 
      // processed DMA request is returned to
      // upward queue
      mem_fetch *mf = pcie_read_latency_queue->mf;

      simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();

      // push the memory fetch into the gmmu to cu queue
      (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);
    }
    pcie_read_latency_queue = NULL;
  }

  // schedule a transfer if there is a pending item in staging queue and
  // nothing is being served at the read latency queue and we have available
  // free pages

  if (!pcie_read_stage_queue.empty() && pcie_read_latency_queue == NULL &&
      m_gpu->get_global_memory()->get_free_pages() >=
          pcie_read_stage_queue.front()->page_list.size()) {

    std::list<pcie_latency_t *>::const_iterator iter =
        pcie_read_stage_queue.begin();
    for (; iter != pcie_read_stage_queue.end(); iter++) {
      if ((*iter)->type == latency_type::DMA) {
        break;
      }
    }

    // prioritize dma before page migration
    if (iter == pcie_read_stage_queue.end()) {
      pcie_read_latency_queue = pcie_read_stage_queue.front();
    } else {
      pcie_read_latency_queue = *iter;
    }

    // when there is a page fault and read data from CPU
    if (pcie_read_latency_queue->type == latency_type::PCIE_READ) {
      //pcie_read_latency_queue->ready_cycle = get_ready_cycle(pcie_read_latency_queue->page_list.size());
      pcie_read_latency_queue->ready_cycle = get_ready_cycle(pcie_read_latency_queue);
      if (new_debug_execution == 1){
        printf("pcie_read_latency_queue->ready_cycle: %lld\n", pcie_read_latency_queue->ready_cycle);
        std::cout <<"---------------------" << std::endl;
      }
      if (sim_prof_enable) {
        event_stats *cp_h2d =
            new memory_stats(memcpy_h2d, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
                             pcie_read_latency_queue->ready_cycle,
                             pcie_read_latency_queue->start_addr,
                             pcie_read_latency_queue->size, 0);
        sim_prof[m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle].push_back(cp_h2d);
      }

      for (unsigned long long read_period = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
           read_period != pcie_read_latency_queue->ready_cycle; read_period++)
        m_new_stats->pcie_read_utilization.push_back(std::make_pair(
            read_period,
            get_pcie_utilization(pcie_read_latency_queue->page_list.size())));

      m_gpu->get_global_memory()->alloc_pages(
          pcie_read_latency_queue->page_list.size());
    } else if (pcie_read_latency_queue->type ==
               latency_type::PAGE_FAULT) { // schedule far-fault for transfer
      // only for schedule 
      // so the latency is fixed to PF latency
      pcie_read_latency_queue->ready_cycle =
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle + m_config.page_fault_latency * pcie_read_latency_queue->page_list.size();

      if (sim_prof_enable) {
        event_stats *mf_fault = new page_fault_stats(
            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            pcie_read_latency_queue->page_list,
            pcie_read_latency_queue->page_list.size() * m_config.page_size);
        fault_stats.push_back(mf_fault);
      }
    } else if (pcie_read_latency_queue->type == latency_type::DMA) { // schedule DMA request for transfer
      pcie_read_latency_queue->ready_cycle =
          get_ready_cycle_dma(pcie_read_latency_queue->mf->get_access_size());
      if (sim_prof_enable) {
        event_stats *ma_dma =
            new memory_stats(dma, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
                             pcie_read_latency_queue->ready_cycle,
                             pcie_read_latency_queue->mf->get_addr(),
                             pcie_read_latency_queue->mf->get_access_size(), 0);
        sim_prof[m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle].push_back(ma_dma);
      }
    }

    // remove the scheduled transfer from read stage queue
    if (iter == pcie_read_stage_queue.end()) {
      pcie_read_stage_queue.pop_front();
    } else {
      pcie_read_stage_queue.erase(iter);
    }
  }

  std::map<mem_addr_t, std::list<mem_fetch *>> page_fault_this_turn;

  // check the page_table_walk_delay_queue
  while (!page_table_walk_queue.empty() &&
         ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
          page_table_walk_queue.front().ready_cycle)) {

    mem_fetch *mf = page_table_walk_queue.front().mf;

    std::list<mem_addr_t> page_list = m_gpu->get_global_memory()->get_faulty_pages(mf->get_addr(), mf->get_access_size());

    simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
    // if there is no page fault, directly return to the upward queue of cluster
    if (page_list.empty()) {
      mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(mf->get_mem_access().get_addr());
      check_write_stage_queue(page_num, false);
      (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);
      m_new_stats->mf_page_hit[simt_cluster_id]++;
      // per pfn
      m_new_stats->page_tlb_miss_pt_hit_times[page_num]++;

    } else {
      assert(page_list.size() == 1);

      mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(mf->get_mem_access().get_addr());
      m_new_stats->mf_page_miss[simt_cluster_id]++;
      // per pfn
      m_new_stats->page_tlb_miss_pt_miss_times[page_num]++;

      // per cycles
      m_new_stats->cycle_pf_times[(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle)/1000].push_back(page_num);

      // the page request is already there in MSHR either as a page fault or as
      // part of scheduled prefetch request
      if (req_info.find(*(page_list.begin())) != req_info.end()) {
        m_new_stats->mf_page_fault_pending++;
        // per pfn
        m_new_stats->page_fault_pending_times[page_num]++;

        req_info[*(page_list.begin())].push_back(mf);
      } else {

        // if the memory fetch is part of any requests in the prefetch command
        // buffer then add it to the incoming replayable_nacks
        std::list<prefetch_req>::iterator iter;

        for (iter = prefetch_req_buffer.begin(); iter != prefetch_req_buffer.end(); iter++) {
          if (iter->start_addr <= mf->get_addr() && mf->get_addr() < iter->start_addr + iter->size) {
            m_new_stats->mf_page_fault_pending++;
            // per pfn
            m_new_stats->page_fault_pending_times[page_num]++;
            iter->incoming_replayable_nacks[page_list.front()].push_back(mf);
            break;
          }
        }

        // if the memory fetch is not part of any request in the prefetch
        // command buffer
        if (iter == prefetch_req_buffer.end()) {

          // if dma is enabled/it is a write access/read access counter hasn't
          // reached thresold
          if (!should_cause_page_migration(mf->get_mem_access().get_addr(),
                                           mf->get_mem_access().get_type() ==
                                               GLOBAL_ACC_W)) {

            m_new_stats->num_dma++;
            pcie_latency_t *p_t = new pcie_latency_t();

            mf->set_dma();

            p_t->mf = mf;
            p_t->type = latency_type::DMA;

            pcie_read_stage_queue.push_back(p_t);
          } else {
            if (dma_mode != dma_type::DISABLED &&
                mf->get_mem_access().get_type() == GLOBAL_ACC_W) {
              m_new_stats->dma_page_transfer_write++;
            } else if (dma_mode != dma_type::DISABLED &&
                       mf->get_mem_access().get_type() == GLOBAL_ACC_R) {
              m_new_stats->dma_page_transfer_read++;
            }
            page_fault_this_turn[page_list.front()].push_back(mf);
          }
        }
      }
    }

    page_table_walk_queue.pop_front();
  }

  // call hardware prefetcher based on the current page faults
  do_hardware_prefetch(page_fault_this_turn);

  // --------------------------------------
  // fetch from cluster's cu to gmmu queue and push it into the page table delay queue
  // suffer PTW latency
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {

    if (!(m_gpu->getSIMTCluster(i))->empty_cu_gmmu_queue()) {

      mem_fetch *mf = (m_gpu->getSIMTCluster(i))->front_cu_gmmu_queue();

      struct page_table_walk_latency_t pt_t;
      pt_t.mf = mf;
      pt_t.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle + m_config.page_table_walk_latency;
      page_table_walk_queue.push_back(pt_t);
      (m_gpu->getSIMTCluster(i))->pop_cu_gmmu_queue();
    }
  }
  // --------------------------------------

  // check if there is an active outstanding prefetch request
  if (!prefetch_req_buffer.empty() && prefetch_req_buffer.front().active) {

    prefetch_req &pre_q = prefetch_req_buffer.front();

    // schedule for page transfers from the active prefetch request when there
    // is no pending transfer for the same can be the very first time or a
    // scheduled big chunk of pages (2MB) is finsihed just now
    if (pre_q.pending_prefetch.empty()) {

      // case when the last schedule finished, it is not the first time
      if (pre_q.cur_addr > pre_q.start_addr) {

        if (sim_prof_enable) {
          update_sim_prof_prefetch_break_down(m_gpu->gpu_sim_cycle +
                                              m_gpu->gpu_tot_sim_cycle);
        }

        m_new_stats->pf_fault_latency.back().second =
            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle -
            m_new_stats->pf_fault_latency.back().second;

        // all the memory fetches created by core on page fault were aggreagted
        // earlier now they are replayed back together to the core
        for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator iter =
                 pre_q.outgoing_replayable_nacks.begin();
             iter != pre_q.outgoing_replayable_nacks.end(); iter++) {

          for (std::list<mem_fetch *>::iterator iter2 = iter->second.begin();
               iter2 != iter->second.end(); iter2++) {

            mem_fetch *mf = *iter2;

            simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
            // push them to the upward queue to replay them back to the
            // corresponding core in bulk
            (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);
          }
        }
        pre_q.outgoing_replayable_nacks.clear();
      }

      // all the memory fetches have been replayed and
      // the prefetch request is completed entirely
      // now signal the stream that the operation is finished so that it can
      // schedule something else
      if (pre_q.cur_addr == pre_q.start_addr + pre_q.size) {

        pre_q.m_stream->record_next_done();

        if (sim_prof_enable) {
          update_sim_prof_prefetch(pre_q.start_addr, pre_q.size,
                                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        }

        prefetch_req_buffer.pop_front();
        return;
      }

      mem_addr_t start_addr = 0;

      pcie_latency_t *p_t = new pcie_latency_t();

      // break the loop if
      //  Case 1: reach the end of this prefetch
      //  Case 2: it reaches the 2MB line from starting of the allocation
      //  Case 3: it encounters a valid page in between
      do {
        // get the page number for the current updated address
        mem_addr_t page_num =
            m_gpu->get_global_memory()->get_page_num(pre_q.cur_addr);

        // update the current address by page size as we break a big chunk (2MB)
        // in the granularity of the smallest unit of page
        pre_q.cur_addr += m_config.page_size;

        // check for Case 3, i.e., we encounter a valid page
        if (m_gpu->get_global_memory()->is_valid(page_num)) {

          m_new_stats->pf_page_hit++;

          // check if this page is currently written back
          check_write_stage_queue(page_num, false);

          // break out of loop only when we have already scheduled some pages
          // for transfer if not we will continue skipping valid pages if any
          // until we find some invalid pages to transfer
          if (!pre_q.pending_prefetch.empty()) {
            break;
          }
        } else {

          m_new_stats->pf_page_miss++;

          // remember this page as pending under the prefetch request
          pre_q.pending_prefetch.push_back(page_num);

          if (start_addr == 0) {
            start_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
            p_t->start_addr = pre_q.cur_addr;
          }

          // just create a placeholder in MSHR for the memory fetches created by
          // core on page fault later in the time so that they go to outgoing
          // replayable nacks, rather than incoming
          req_info[page_num];

          // incoming nacks hold the list of page faults for the transfer which
          // has not been scheduled yet so instead of pushing them to MSHR and
          // then again getting back to the outgoing list directly switch
          // between the incoming and outgoing list of replayable nacks
          if (pre_q.incoming_replayable_nacks.find(page_num) !=
              pre_q.incoming_replayable_nacks.end()) {
            pre_q.outgoing_replayable_nacks[page_num].merge(
                pre_q.incoming_replayable_nacks[page_num]);
            pre_q.incoming_replayable_nacks.erase(page_num);
          }

          // schedule this page as it is not valid to the read stage queue
          p_t->page_list.push_back(page_num);
          m_new_stats->pf_page_fault_latency[page_num].push_back(
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        }

      } while (
          pre_q.cur_addr !=
              (pre_q.start_addr +
               pre_q.size) && // check for Case 1, i.e., we reached the end of
                              // prefetch request
          ((unsigned long long)(pre_q.cur_addr - pre_q.allocation_addr)) %
              ((unsigned long long)
                   MAX_PREFETCH_SIZE)); // Case 2: allowing maximum transfer
                                        // size as huge page size of 2MB

      if (!p_t->page_list.empty()) {
        p_t->size = p_t->page_list.size() * m_config.page_size;
        p_t->type = latency_type::PCIE_READ;
        pcie_read_stage_queue.push_back(p_t);
      }

      m_new_stats->pf_fault_latency.push_back(
          std::make_pair(pre_q.pending_prefetch.size() * m_config.page_size,
                         m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle));

      if (sim_prof_enable && !pre_q.pending_prefetch.empty()) {
        event_stats *cp_pref_bd = new memory_stats(
            prefetch_breakdown, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, start_addr,
            pre_q.pending_prefetch.size() * m_config.page_size,
            pre_q.m_stream->get_uid());
        sim_prof[m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle].push_back(cp_pref_bd);
      }
    }
  }
  
  if (!skip_cycles && !all_warps.empty() && all_warps.size() == fail_warps.size()) {
    std::set<int> temp_set;
    for (std::list<shd_warp_t *>::iterator iter = fail_warps.begin(); iter != fail_warps.end(); iter++) {
      temp_set.insert((*iter)->get_warp_id());
    }

    for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator iter=req_info.begin();
          iter != req_info.end(); ++iter) {
      for(std::list<mem_fetch *>::iterator iter2=iter->second.begin(); iter2!=iter->second.end(); ++iter2) {
        if (temp_set.find((*iter2)->get_inst().warp_id()) != temp_set.end())
          temp_set.erase(temp_set.find((*iter2)->get_inst().warp_id()));
      }
    }

    if (temp_set.empty()) {
      skip_cycles = true;
    } else {
      skip_cycles = false;
    }
    fflush(stdout);
  }
}
// -------------------------------------------------------------------------
void gmmu_t::do_hardware_prefetch( std::map<mem_addr_t, std::list<mem_fetch *>> &page_fault_this_turn) {
  // now decide on transfers as a group of page faults and prefetches
  if (!page_fault_this_turn.empty()) {
    unsigned long long num_pages_read_stage_queue = 0;

    for (std::list<pcie_latency_t *>::iterator iter =
             pcie_read_stage_queue.begin();
         iter != pcie_read_stage_queue.end(); iter++) {
      num_pages_read_stage_queue += (*iter)->page_list.size();
    }

    std::list<std::list<mem_addr_t>> all_transfer_all_page;
    std::list<std::list<mem_addr_t>> all_transfer_faulty_pages;
    std::map<mem_addr_t, std::list<mem_fetch *>> temp_req_info;

    // create a tree structure large page -> basic blocks -> faulty pages
    std::map<mem_addr_t, std::map<mem_addr_t, std::list<mem_addr_t>>> block_tree;

    if (prefetcher == hwardware_prefetcher::DISBALED || prefetcher == hwardware_prefetcher::RANDOM) {
      for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator it = page_fault_this_turn.begin(); it != page_fault_this_turn.end(); it++) {
        std::list<mem_addr_t> temp_pages;
        temp_pages.push_back(it->first);

        mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(it->first);
        
        struct lp_tree_node *root = get_lp_node(page_addr);
        update_basic_block(root, page_addr, m_config.page_size, true);

        all_transfer_all_page.push_back(temp_pages);
        all_transfer_faulty_pages.push_back(temp_pages);

        temp_req_info[it->first];

        if (prefetcher == hwardware_prefetcher::RANDOM) {
          struct lp_tree_node *root =
              get_lp_node(m_gpu->get_global_memory()->get_mem_addr(it->first));

          size_t random_size =
              (rand() % (root->size / m_config.page_size)) * m_config.page_size;

          if (random_size > root->size) {
            random_size -= root->size;
          }

          mem_addr_t prefetch_addr = root->addr + random_size;

          mem_addr_t prefetch_page_num =
              m_gpu->get_global_memory()->get_page_num(prefetch_addr);

          if (!m_gpu->get_global_memory()->is_valid(prefetch_page_num) &&
              page_fault_this_turn.find(prefetch_addr) ==
                  page_fault_this_turn.end() &&
              temp_req_info.find(prefetch_page_num) == temp_req_info.end() &&
              req_info.find(prefetch_page_num) == req_info.end()) {

            mem_addr_t page_addr =
                m_gpu->get_global_memory()->get_mem_addr(prefetch_page_num);
            struct lp_tree_node *root = get_lp_node(page_addr);
            update_basic_block(root, page_addr, m_config.page_size, true);

            all_transfer_all_page.back().push_back(prefetch_page_num);

            temp_req_info[prefetch_page_num];
          }
        }
      }
    } else {
      std::map<mem_addr_t, std::set<mem_addr_t>> lp_pf_groups;

      for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator it =
               page_fault_this_turn.begin();
           it != page_fault_this_turn.end(); it++) {
        mem_addr_t page_addr =
            m_gpu->get_global_memory()->get_mem_addr(it->first);

        struct lp_tree_node *root = get_lp_node(page_addr);

        lp_pf_groups[root->addr].insert(page_addr);
      }

      for (std::map<mem_addr_t, std::set<mem_addr_t>>::iterator lp_pf_iter = lp_pf_groups.begin();
           lp_pf_iter != lp_pf_groups.end(); lp_pf_iter++) {
        std::set<mem_addr_t> schedulable_basic_blocks;

        // list of all invalid pages and pages with fault from all basic blocks
        // to satisfy current transfer size
        std::list<mem_addr_t> cur_transfer_all_pages;
        std::list<mem_addr_t> cur_transfer_faulty_pages;

        for (std::set<mem_addr_t>::iterator pf_iter =
                 lp_pf_iter->second.begin();
             pf_iter != lp_pf_iter->second.end(); pf_iter++) {
          mem_addr_t page_addr = *pf_iter;

          struct lp_tree_node *root = get_lp_node(page_addr);

          mem_addr_t bb_addr =
              update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, true);

          schedulable_basic_blocks.insert(bb_addr);

          cur_transfer_faulty_pages.push_back(
              m_gpu->get_global_memory()->get_page_num(page_addr));
        }

        if (prefetcher == hwardware_prefetcher::TBN) {
          struct lp_tree_node *root = get_lp_node(lp_pf_iter->first);
          traverse_and_fill_lp_tree(root, schedulable_basic_blocks);
        }

        for (std::set<mem_addr_t>::iterator bb =
                 schedulable_basic_blocks.begin();
             bb != schedulable_basic_blocks.end(); bb++) {

          block_access_list.push_back(
              std::make_pair(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, *bb));

          // all the invalid pages in the current 64 K basic block of transfer
          std::list<mem_addr_t> all_block_pages =
              m_gpu->get_global_memory()->get_faulty_pages(*bb,
                                                           MIN_PREFETCH_SIZE);

          for (std::list<mem_addr_t>::iterator pg_iter =
                   all_block_pages.begin();
               pg_iter != all_block_pages.end(); pg_iter++) {
            if (temp_req_info.find(*pg_iter) == temp_req_info.end()) {
              // mark entry into mshr for all pages in the current basic block
              temp_req_info[*pg_iter];
              cur_transfer_all_pages.push_back(*pg_iter);
            }
          }
        }

        all_transfer_all_page.push_back(cur_transfer_all_pages);
        all_transfer_faulty_pages.push_back(cur_transfer_faulty_pages);
      }
    }

    for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator iter =
             temp_req_info.begin();
         iter != temp_req_info.end(); iter++) {
      req_info[iter->first];
      req_info[iter->first].merge(iter->second);
    }

    std::list<std::list<mem_addr_t>>::iterator all_pg_iter =
        all_transfer_all_page.begin();
    std::list<std::list<mem_addr_t>>::iterator all_pf_iter =
        all_transfer_faulty_pages.begin();

    for (; all_pg_iter != all_transfer_all_page.end();
         all_pg_iter++, all_pf_iter++) {
      // now we found all the basic blocks for the current transfer size
      // we now decide on the splits based on page faults
      std::list<mem_addr_t>::iterator pf_iter = all_pf_iter->begin();
      std::list<mem_addr_t>::iterator pg_iter = all_pg_iter->begin();

      std::list<mem_addr_t>::iterator prev_pg_iter;

      while (pg_iter != all_pg_iter->end()) {

        // if there is a gap between current and last page
        // it can be if two basic blocks selected for current transfer size
        // is separated by other basic blocks
        // then we send this basic block (or remaining of so) for transfer
        if (pg_iter != all_pg_iter->begin()) {
          prev_pg_iter = pg_iter;
          --prev_pg_iter;

          if ((*pg_iter) != ((*prev_pg_iter) + 1)) {

            // add the current split for transfer
            pcie_latency_t *p_t = new pcie_latency_t();
            p_t->start_addr =
                m_gpu->get_global_memory()->get_mem_addr(all_pg_iter->front());
            p_t->page_list =
                std::list<mem_addr_t>(all_pg_iter->begin(), pg_iter);
            p_t->size = p_t->page_list.size() * m_config.page_size;
            p_t->type = latency_type::PCIE_READ;

            pcie_read_stage_queue.push_back(p_t);

            // remove the scheduled pages from all pages and move the pointer
            pg_iter = all_pg_iter->erase(all_pg_iter->begin(), pg_iter);
          }
        }

        // we found a page on which a page fault request is pending
        // now we split upto this and create a memory transfer
        if ((pf_iter != all_pf_iter->end()) && ((*pf_iter) == (*pg_iter))) {

          if (m_config.enable_accurate_simulation) {
            pcie_latency_t *f_t = new pcie_latency_t();
            f_t->page_list.push_back(*pf_iter);
            f_t->type = latency_type::PAGE_FAULT;
            pcie_read_stage_queue.push_back(f_t);
          }

          // add the current split for transfer
          pcie_latency_t *p_t = new pcie_latency_t();
          p_t->start_addr =
              m_gpu->get_global_memory()->get_mem_addr(all_pg_iter->front());
          p_t->page_list =
              std::list<mem_addr_t>(all_pg_iter->begin(), ++pg_iter);
          p_t->size = p_t->page_list.size() * m_config.page_size;
          p_t->type = latency_type::PCIE_READ;

          pcie_read_stage_queue.push_back(p_t);

          // remove the scheduled pages from all pages and move the pointer
          pg_iter = all_pg_iter->erase(all_pg_iter->begin(), pg_iter);
          pf_iter++;
        } else {
          pg_iter++;
        }
      }

      // prefetch the remaining from the 64K basic block
      if (!all_pg_iter->empty()) {
        pcie_latency_t *p_t = new pcie_latency_t();
        p_t->start_addr =
            m_gpu->get_global_memory()->get_mem_addr(all_pg_iter->front());
        p_t->page_list = *all_pg_iter;
        p_t->size = p_t->page_list.size() * m_config.page_size;
        p_t->type = latency_type::PCIE_READ;

        pcie_read_stage_queue.push_back(p_t);
      }
    }

    // adding statistics for prefetch
    for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator iter2 = page_fault_this_turn.begin(); iter2 != page_fault_this_turn.end(); iter2++) {
      assert(req_info[iter2->first].size() == 0);

      // add the pending prefecthes to the MSHR entry
      req_info[iter2->first] = iter2->second;

      m_new_stats->mf_page_fault_outstanding++;

      // per pfn
      m_new_stats->page_fault_outstand_times[iter2->first]++;

      m_new_stats->mf_page_fault_pending += req_info[iter2->first].size() - 1;

      m_new_stats->mf_page_fault_latency[iter2->first].push_back(m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    }

    if (!over_sub && m_gpu->get_global_memory()->should_evict_page(
                         num_pages_read_stage_queue + temp_req_info.size(), 0,
                         m_config.free_page_buffer_percentage)) {

      if (m_config.enable_smart_runtime) {
        update_memory_management_policy();
      } else {
        update_hardware_prefetcher_oversubscribed();
      }

      over_sub = true;
    }
  }
}

gmmu_t *gpgpu_sim::getGmmu() { return m_gmmu; }

// uvm: ==============================================

gpgpu_new_stats::gpgpu_new_stats(const gpgpu_sim_config &config)
    : m_config(config) {
  tlb_hit = new unsigned long long[m_config.num_cluster()];
  tlb_miss = new unsigned long long[m_config.num_cluster()];
  tlb_val = new unsigned long long[m_config.num_cluster()];
  tlb_evict = new unsigned long long[m_config.num_cluster()];
  tlb_page_evict = new unsigned long long[m_config.num_cluster()];

  mf_page_hit = new unsigned long long[m_config.num_cluster()];
  mf_page_miss = new unsigned long long[m_config.num_cluster()];

  mf_page_fault_outstanding = 0;
  mf_page_fault_pending = 0;

  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    tlb_hit[i] = 0;
    tlb_miss[i] = 0;
    tlb_val[i] = 0;
    tlb_evict[i] = 0;
    tlb_page_evict[i] = 0;
    mf_page_hit[i] = 0;
    mf_page_miss[i] = 0;
  }

  pf_page_hit = 0;
  pf_page_miss = 0;

  page_evict_not_dirty = 0;
  page_evict_dirty = 0;

  num_dma = 0;
  dma_page_transfer_read = 0;
  dma_page_transfer_write = 0;

  tlb_thrashing =
      new std::map<mem_addr_t, std::vector<bool>>[m_config.num_cluster()];

  ma_latency =
      new std::map<unsigned,
                   std::pair<bool, unsigned long long>>[m_config.num_cluster()];

  page_access_times =
      new std::map<mem_addr_t, unsigned>[m_config.num_cluster()];
}

void gpgpu_new_stats::print_pcie(FILE *fout) const {
  fprintf(fout, "Read lanes:\n");
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_read_utilization.begin();
       iter != pcie_read_utilization.end(); iter++) {
    fprintf(fout, "%llu %f\n", iter->first, iter->second);
  }
  fprintf(fout, "Write lanes:\n");
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_write_utilization.begin();
       iter != pcie_write_utilization.end(); iter++) {
    fprintf(fout, "%llu %f\n", iter->first, iter->second);
  }
}

void gpgpu_new_stats::print_access_pattern_detail(FILE *fout) const {
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(fout, "Shader %u\n", i);
    for (std::map<mem_addr_t, unsigned>::const_iterator iter =
             page_access_times[i].begin();
         iter != page_access_times[i].end(); iter++) {
      fprintf(fout, "%u %u\n", iter->first, iter->second);
    }
  }
}

// Figure 1, 6
void gpgpu_new_stats::print_access_pattern(FILE *fout) {
  // PFN Times
  std::map<mem_addr_t, unsigned> tot_access;

  fprintf(fout, "pfn,tot_access_count,tlb_hit,tlb_tot_miss,tlb_miss_pt_hit,tlb_miss_pt_miss,tlb_miss_pt_miss_outstand,tlb_miss_pt_miss_pending\n");
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    for (std::map<mem_addr_t, unsigned>::const_iterator iter = page_access_times[i].begin(); iter != page_access_times[i].end(); iter++) {
      tot_access[iter->first] += iter->second;
    }
  }

  // tot_access = tlb_hit + tlb_miss
  // tlb_miss = pt_hit + pt_miss
  // pt_miss = pt_out + pt_pend
  // breakdown of per_pfn_page_access
  for (auto iter = tot_access.begin(); iter != tot_access.end(); ++iter) {
      fprintf(fout, "%llu,%u,%u,%u,%u,%u,%u,%u\n", 
          iter->first,                              // pfn
          iter->second,                             // tot_access
          page_tlb_hit_times[iter->first],          // tlb_hit when access this page
          page_tlb_miss_times[iter->first],         // tlb_miss when access this page
          page_tlb_miss_pt_hit_times[iter->first],  // tlb_miss but page table hit
          page_tlb_miss_pt_miss_times[iter->first], // tlb_miss and pt_miss
          page_fault_outstand_times[iter->first],   // tlb_miss and pt_miss and the first time PF
          page_fault_pending_times[iter->first]     // tlb_miss and pt_miss and pending PF
      );
  }
}
// Figure 2
void gpgpu_new_stats::print_time_and_access(FILE *fout) const {
  // PFN(Addr>>PGSize) MemAddr Size Cycle R/W SM Warp
  for (std::list<access_info>::const_iterator iter = time_and_page_access.begin(); iter != time_and_page_access.end(); iter++) {
    fprintf(fout, "%u,0x%x,%u,%llu,%d,%u,%u\n", 
            iter->page_no, 
            iter->mem_addr,
            iter->size, 
            iter->cycle, 
            iter->is_read, 
            iter->sm_id, 
            iter->warp_id);
  }
  // // Kernel Start End
  // for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter = sim_prof.begin(); iter != sim_prof.end(); iter++) {
  //   for (std::list<event_stats *>::iterator iter2 = iter->second.begin();iter2 != iter->second.end(); iter2++) {
  //     if ((*iter2)->type == kernel_launch) {
  //       fprintf(fout, "K: %llu %llu\n", (*iter2)->start_time,
  //               (*iter2)->end_time);
  //     }
  //   }
  // }
}


// Figure 3, 5
// cycles how pf and pe numbers change
void gpgpu_new_stats::print_cycle_pf_nums(FILE *fout) {
    fprintf(fout, "cycle-1000, n_pf\n");
    fprintf(fout, "pfns\n");
    
    for (auto iter = cycle_pf_times.begin(); iter != cycle_pf_times.end(); ++iter) {
        fprintf(fout, "%llu, %u\n", iter->first, iter->second.size());
        // Iterate through the vector of mem_addr_t
        for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2) {
            // Print the element
            fprintf(fout, "%llu", *iter2);
            // Print a comma if it's not the last element
            if (iter2 + 1 != iter->second.end()) {
                fprintf(fout, ",");
            }
        }
        fprintf(fout, "\n");
    }


}

  // Figure 3, 5
void gpgpu_new_stats::print_cycle_pe_nums(FILE *fout){
    fprintf(fout, "cycle-1000, n_pe\n");
    fprintf(fout, "pfns\n");
    
    if(cycle_pe_times.size()!=0){
      for (auto iter = cycle_pe_times.begin(); iter != cycle_pe_times.end(); ++iter) {
          fprintf(fout, "%llu, %u\n", iter->first, iter->second.size());
          // Iterate through the vector of mem_addr_t
          for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2) {
              // Print the element
              fprintf(fout, "%llu", *iter2);
              // Print a comma if it's not the last element
              if (iter2 + 1 != iter->second.end()) {
                  fprintf(fout, ",");
              }
          }
          fprintf(fout, "\n");
      }
    }


}

// Figure 4
void gpgpu_new_stats::print_cycle_pageset(FILE*fout){
  
  fprintf(fout,"cycle-1000,n_pages\n");
  fprintf(fout,"pfn\n");
  for (auto iter = cycle_pages_valid.begin();iter!=cycle_pages_valid.end();++iter){
    unsigned long long cycle = iter->first;
    // how many pages
    unsigned valid_pages_num = iter->second.size();
    fprintf(fout,"%llu,%u\n", cycle, valid_pages_num);

    bool first = true;
    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2) {
        if (!first) {
            fprintf(fout, ",");
        }
        fprintf(fout, "%llu", *iter2);
        first = false;
    }
    
    fprintf(fout,"\n");
  }
}


// Figure 7
void gpgpu_new_stats::print_page_access_stats(FILE *fout) const{
  // tlb_hit
  unsigned long long tot_tlb_hit = 0;

  // tlb_miss_pf_out
  unsigned long long tlb_miss_pt_miss_out = mf_page_fault_outstanding;  
  // tlb_miss_pf_pend
  unsigned long long tlb_miss_pt_miss_pend = mf_page_fault_pending;  

  unsigned long long tot_page_hit = 0;

  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    tot_page_hit += mf_page_hit[i];
    tot_tlb_hit += tlb_hit[i];
  }
  fprintf(fout,"tlb_hit,pt_hit,pt_miss_outstand,pt_miss_pend\n");
  fprintf(fout, "%llu,%llu,%llu,%llu\n", tot_tlb_hit, tot_page_hit, tlb_miss_pt_miss_out, tlb_miss_pt_miss_pend);

}

// FIgure 9
void gpgpu_new_stats::print_page_read_write(FILE *fout){

  fprintf(fout,"pfn,n_read,n_write\n");
  for (auto iter = page_rw_times.begin();iter!=page_rw_times.end();iter++){
    mem_addr_t page_no = iter->first;
    unsigned reads = iter->second.first;
    unsigned writes = iter->second.second;
    fprintf(fout,"%llu,%u,%u\n", page_no, reads, writes);
  }

}



// print all statistics about uvm
void gpgpu_new_stats::print(FILE *fout) const {
  fprintf(fout, "========================================UVM "
                "statistics==============================\n");

  fprintf(fout, "========================================TLB "
                "statistics(access)==============================\n");
  unsigned long long tot_tlb_hit = 0;
  unsigned long long tot_tlb_miss = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(fout,
            "Shader%u: Tlb_access: %llu Tlb_hit: %llu Tlb_miss: %llu "
            "Tlb_hit_rate: %f\n",
            i, tlb_hit[i] + tlb_miss[i], tlb_hit[i], tlb_miss[i],
            ((float)tlb_hit[i]) / ((float)(tlb_hit[i] + tlb_miss[i])));
    tot_tlb_hit += tlb_hit[i];
    tot_tlb_miss += tlb_miss[i];
  }

  fprintf(fout,
          "Tlb_tot_access: %llu Tlb_tot_hit: %llu, Tlb_tot_miss: %llu, "
          "Tlb_tot_hit_rate: %f\n",
          tot_tlb_hit + tot_tlb_miss, tot_tlb_hit, tot_tlb_miss,
          ((float)tot_tlb_hit) / ((float)(tot_tlb_hit + tot_tlb_miss)));

  fprintf(fout, "========================================TLB "
                "statistics(validate)==============================\n");
  unsigned long long tot_tlb_val = 0;
  unsigned long long tot_tlb_inval_te = 0;
  unsigned long long tot_tlb_inval_pe = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(fout,
            "Shader%u: Tlb_validate: %llu Tlb_invalidate: %llu Tlb_evict: %llu "
            "Tlb_page_evict: %llu\n",
            i, tlb_val[i], tlb_evict[i] + tlb_page_evict[i], tlb_evict[i],
            tlb_page_evict[i]);
    tot_tlb_val += tlb_val[i];
    tot_tlb_inval_te += tlb_evict[i];
    tot_tlb_inval_pe += tlb_page_evict[i];
  }

  fprintf(fout,
          "Tlb_tot_valiate: %llu Tlb_invalidate: %llu, Tlb_tot_evict: %llu, "
          "Tlb_tot_evict page: %llu\n",
          tot_tlb_val, tot_tlb_inval_te + tot_tlb_inval_pe, tot_tlb_inval_te,
          tot_tlb_inval_pe);

  fprintf(fout, "========================================TLB "
                "statistics(thrashing)==============================\n");
  std::map<mem_addr_t, unsigned> tlb_thrash[m_config.num_cluster()];
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    for (std::map<mem_addr_t, std::vector<bool>>::const_iterator iter =
             tlb_thrashing[i].begin();
         iter != tlb_thrashing[i].end(); iter++) {
      for (unsigned j = 0; j != iter->second.size(); j++) {
        if (j + 2 >= iter->second.size())
          break;
        if (iter->second[j] == true && iter->second[j + 1] == false &&
            iter->second[j + 2] == true)
          tlb_thrash[i][iter->first]++;
      }
    }
  }

  unsigned tot_tlb_thrash = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    unsigned s_thrash = 0;
    fprintf(fout, "Shader%u: ", i);
    for (std::map<mem_addr_t, unsigned>::iterator iter = tlb_thrash[i].begin();
         iter != tlb_thrash[i].end(); iter++) {
      fprintf(fout, "Page: %u Trashed: %u | ", iter->first, iter->second);
      s_thrash += iter->second;
    }
    fprintf(fout, "Total %u\n", s_thrash);
    tot_tlb_thrash += s_thrash;
  }
  fprintf(fout, "Tlb_tot_thrash: %u\n", tot_tlb_thrash);

  fprintf(fout, "========================================Page fault "
                "statistics==============================\n");

  unsigned long long tot_page_hit = 0;
  unsigned long long tot_page_miss = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(
        fout,
        "Shader%u: Page_table_access:%llu Page_hit: %llu Page_miss: %llu "
        "Page_hit_rate: %f\n",
        i, 
        mf_page_hit[i] + mf_page_miss[i], 
        mf_page_hit[i], 
        mf_page_miss[i],
        ((float)mf_page_hit[i]) / ((float)(mf_page_hit[i] + mf_page_miss[i])));
    tot_page_hit += mf_page_hit[i];
    tot_page_miss += mf_page_miss[i];
  }

  fprintf(fout,
          "Page_table_tot_access: %llu Page_tot_hit: %llu, Page_tot_miss %llu, "
          "Page_tot_hit_rate: %f Page_tot_fault: %llu Page_tot_pending: %llu\n",
          tot_page_hit + tot_page_miss, 
          tot_page_hit, 
          tot_page_miss,
          ((float)tot_page_hit) / ((float)(tot_page_hit + tot_page_miss)),
          mf_page_fault_outstanding, 
          mf_page_fault_pending);

  float avg_mf_latency = 0;
  unsigned long long tot_mf_fault = 0;
  for (std::map<mem_addr_t, std::list<unsigned long long>>::const_iterator iter = mf_page_fault_latency.begin(); iter != mf_page_fault_latency.end(); iter++) {
    for (std::list<unsigned long long>::const_iterator iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++) {
      avg_mf_latency =
          ((float)tot_mf_fault) / ((float)(tot_mf_fault + 1)) * avg_mf_latency +
          ((float)(*iter2)) / ((float)(tot_mf_fault + 1));
      tot_mf_fault++;
    }
  }
  fprintf(fout, "Total_memory_access_page_fault: %llu, Average_latency: %f, "
                "skipped cycles: %llu, Total_skipped_cycles: %llu \n",
                tot_mf_fault, avg_mf_latency, skipped_cycles, gpu_tot_skipped_cycle);

  fprintf(fout, "========================================Page thrashing "
                "statistics==============================\n");

  unsigned long long tot_validate = 0;
  for (std::map<mem_addr_t, std::vector<bool>>::const_iterator iter = page_thrashing.begin(); iter != page_thrashing.end(); iter++) {
    for (std::vector<bool>::const_iterator iter2 = iter->second.begin(); iter2 != iter->second.end(); iter2++) {
      if (*iter2 == true)
        tot_validate++;
    }
  }

  fprintf(
      fout,
      "Page_validate: %llu Page_evict_dirty: %llu Page_evict_not_dirty: %llu\n",
      tot_validate, page_evict_dirty, page_evict_not_dirty);

  std::map<mem_addr_t, unsigned> page_thrash;
  for (std::map<mem_addr_t, std::vector<bool>>::const_iterator iter =
           page_thrashing.begin();
       iter != page_thrashing.end(); iter++) {
    for (unsigned j = 0; j != iter->second.size(); j++) {
      if (j + 2 >= iter->second.size())
        break;
      if (iter->second[j] == true && iter->second[j + 1] == false &&
          iter->second[j + 2] == true)
        page_thrash[iter->first]++;
    }
  }

  unsigned tot_page_thrash = 0;
  for (std::map<mem_addr_t, unsigned>::iterator iter = page_thrash.begin();
       iter != page_thrash.end(); iter++) {
    fprintf(fout, "Page: %u Thrashed: %u\n", iter->first, iter->second);
    tot_page_thrash += iter->second;
  }
  fprintf(fout, "Page_tot_thrash: %u\n", tot_page_thrash);

  fprintf(fout, "========================================Memory access "
                "statistics==============================\n");

  /*
     unsigned long long* ma_num = new unsigned long
     long[m_config.num_cluster()]; float* avg_ma_latency = new
     float[m_config.num_cluster()];

     unsigned long long tot_ma_num = 0;
     float tot_avg_ma_latency = 0;

     for(unsigned i = 0; i < m_config.num_cluster(); i++) {
         ma_num[i] = 0;
         avg_ma_latency[i] = 0;
         for(std::map<unsigned, std::pair<bool, unsigned long long>
     >::const_iterator iter = ma_latency[i].begin(); iter !=
     ma_latency[i].end(); iter++) { assert(iter->second.first);
             avg_ma_latency[i] = ((float)ma_num[i]) / ((float)(ma_num[i]+1)) *
     avg_ma_latency[i] + ((float)(iter->second.second)) /
     ((float)(ma_num[i]+1)); ma_num[i]++;
         }
         fprintf(fout, "Shader%u: Memory_access: %u, Avg_memory_access_latency:
     %llu\n", i, ma_latency[i].size(), ((unsigned long long)
     (avg_ma_latency[i])));
     }

     for(unsigned i = 0; i < m_config.num_cluster(); i++) {
         tot_avg_ma_latency = ((float)tot_ma_num) /
     ((float)(tot_ma_num+ma_num[i])) * tot_avg_ma_latency + avg_ma_latency[i] /
     ((float)(tot_ma_num+ma_num[i])) * ((float)ma_num[i]); tot_ma_num +=
     ma_num[i];
     }
     fprintf(fout,"Tot_memory_access: %u, Tot_avg_memory_access_latency:
     %llu\n", tot_ma_num, ((unsigned long long)tot_avg_ma_latency));

     delete[] ma_num;
     delete[] avg_ma_latency;
  */
  fprintf(fout, "========================================Prefetch "
                "statistics==============================\n");

  fprintf(fout,
          "Tot_page_hit: %llu, Tot_page_miss: %llu, Tot_page_fault: %lu\n",
          pf_page_hit, pf_page_miss, pf_fault_latency.size());

  float avg_pf_latency = 0;
  float avg_pref_size = 0;
  float avg_pref_latency = 0;

  unsigned long long tot_pf_fault = 0;
  unsigned long long tot_pref_fault = 0;
  for (std::map<mem_addr_t, std::list<unsigned long long>>::const_iterator
           iter = pf_page_fault_latency.begin();
       iter != pf_page_fault_latency.end(); iter++) {
    for (std::list<unsigned long long>::const_iterator iter2 =
             iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      avg_pf_latency =
          ((float)tot_pf_fault) / ((float)(tot_pf_fault + 1)) * avg_pf_latency +
          ((float)(*iter2)) / ((float)(tot_pf_fault + 1));
      tot_pf_fault++;
    }
  }

  for (std::vector<std::pair<unsigned long, unsigned long long>>::const_iterator
           iter = pf_fault_latency.begin();
       iter != pf_fault_latency.end(); iter++) {
    avg_pref_size += iter->first;
    avg_pref_latency = ((float)tot_pref_fault) / ((float)(tot_pref_fault + 1)) *
                           avg_pref_latency +
                       ((float)(iter->second)) / ((float)(tot_pref_fault + 1));
    tot_pref_fault++;
  }

  avg_pref_size /= ((float)pf_fault_latency.size());
  fprintf(
      fout,
      "Avg_page_latency: %f, Avg_prefetch_size: %f, Avg_prefetch_latency: %f\n",
      avg_pf_latency, avg_pref_size, avg_pref_latency);

  fprintf(fout, "========================================Rdma "
                "statistics==============================\n");
  fprintf(fout, "dma_read: %llu\n", num_dma);
  fprintf(fout, "dma_migration_read %llu\n", dma_page_transfer_read);
  fprintf(fout, "dma_migration_write %llu\n", dma_page_transfer_write);

  fprintf(fout, "========================================PCI-e "
                "statistics==============================\n");
  float avg_r = 0;
  unsigned long long r_0 = 0;
  unsigned long long r_25 = 0;
  unsigned long long r_50 = 0;
  unsigned long long r_75 = 0;
  unsigned long long r_tot = 0;
  float avg_w = 0;
  unsigned long long w_0 = 0;
  unsigned long long w_25 = 0;
  unsigned long long w_50 = 0;
  unsigned long long w_75 = 0;
  unsigned long long w_tot = 0;
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_read_utilization.begin();
       iter != pcie_read_utilization.end(); iter++) {
    if (iter->second <= 0.25) {
      r_0++;
    } else if (iter->second <= 0.5) {
      r_25++;
    } else if (iter->second <= 0.75) {
      r_50++;
    } else {
      r_75++;
    }
    avg_r = (avg_r * ((float)r_tot) + iter->second) / ((float)(r_tot + 1));
    r_tot++;
  }
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_write_utilization.begin();
       iter != pcie_write_utilization.end(); iter++) {
    if (iter->second <= 0.25) {
      w_0++;
    } else if (iter->second <= 0.5) {
      w_25++;
    } else if (iter->second <= 0.75) {
      w_50++;
    } else {
      w_75++;
    }
    avg_w = (avg_w * ((float)w_tot) + iter->second) / ((float)(w_tot + 1));
    w_tot++;
  }

  fprintf(fout, "Pcie_read_utilization: %f\n", avg_r);
  fprintf(fout, "[0-25]: %f, [26-50]: %f, [51-75]: %f, [76-100]: %f\n",
          ((float)r_0) / ((float)r_tot), ((float)r_25) / ((float)r_tot),
          ((float)r_50) / ((float)r_tot), ((float)r_75) / ((float)r_tot));
  fprintf(fout, "Pcie_write_utilization: %f\n", avg_w);
  fprintf(fout, "[0-25]: %f, [26-50]: %f, [51-75]: %f, [76-100]: %f\n",
          ((float)w_0) / ((float)w_tot), ((float)w_25) / ((float)w_tot),
          ((float)w_50) / ((float)w_tot), ((float)w_75) / ((float)w_tot));
}

gpgpu_new_stats::~gpgpu_new_stats() {
  delete[] tlb_hit;
  delete[] tlb_miss;
  delete[] tlb_val;
  delete[] tlb_evict;
  delete[] tlb_page_evict;
  delete[] mf_page_hit;
  delete[] mf_page_miss;
  delete[] page_access_times;
  delete[] tlb_thrashing;
  delete[] ma_latency;
}
