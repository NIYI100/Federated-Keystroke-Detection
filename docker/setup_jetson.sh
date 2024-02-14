# Python 3.6, torch 1.10.0, numpy 1.17.3, Ubuntu 18.04

wget -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl 'https://public.boxcloud.com/d/1/b1!LfQGu8aZ-W2d0epst5gZLKQwabmRuXITNNsUpOEWupvSZq5vClegvaed3pdQMxM3k9yFhV9oSmMzvHg6ascSFsbl6_Vh97X0Pr0xqhjdr-bDlfG44eT-OuGe5WijBQavWjEMgbeYsJhsNhQb6UyNyb9tz6Q8nweieIb3Ov8iRTWaJ9pztVajHo4DxMfDnmoNr8sRT3d5Tb-H8v88ep2PL5OIgCN59ZZcu88aCYrQunMOXnS5XGDYKID0wtGE1DWVM6huaaQQqXqQXWqsjMtZ57mfYoaFQuwRumVzKjhByZJfIJUIwAwZTBjGgjOfsF6aoCV2wTkBn2LmrviA5VBqu2sH3Dv6btszv4Q8qBC4-nEGkY7dXa0eigmuFta9G_9SRGZBXmj-YuQKEibo-QHUqQTdxQVMEuXgmVbq16jxrH0y-dxK_drwHDWnaqz8ecQ6YBHAFDTGhYQpXqtpOej5VhUHIP-ejxY6RYH2gPCzM-x0DSO8DRK0fQXb6u-hew1JMe2QryUuc_fXynWpRg46aue3bM-7ovv2czN93bp6Thh95cTOjwmcepzVQV7LpTQ34sIMjRHHCxv1rp5QQFiGpsWzINsdV29jMyVw9BNBvLsWubAAlwem0Hvag9OXdW7mFWnJEw0a8nRZad-uo_fQdcbKGRUMpAhykY2F811rPMNnGHn_JvD-aMn5GVYBz1iFhJaKGiHANCWJK4yw42TJn4cUETFnCoT1skO9durcJFnebDE2CJr-Qb5ttq7brap9GJ9KkTzKDUlsoqv6eYHEqSdvlCsB1_RUXzOfPmlZrVqlLy1R_qUQl-7CvCQCfBQT2QEBQGRhY0cIZgYnALCZQW7EHPUgRHspK5ojLhRYZEqTgC459IuXyDQH-bKMS8cxuiVwODKtMRGgDVMMi2Q-rYO2WLVIgk-t3Ikr7cH0JjzRGCcGoQOWdmPUZFE_LY2o4zVWvYhhyQP9iN7m_cWmTmr1lSapgTa05JPzOXA5mACYHQIXWufqk8zs3BjDMcH1FIz5WmtCw7wpWLCNbvZdegAoNUpUGCb7nC75yW9hEehQsEZ1ypAX3DvTSoYZxCtIl1Arge0U4sakA6k0Ghq-Vv4poGz-1okeOOS2nqbJmefex9eXgG2LlQBxIFCfLuWRh4NLQcyQv1uepfUBV92ZMLtsMBK9knZfQTwMmQ0MD6Iv_S_t6P5oK85Rmrux07KlCZmHc6D0s2HzP-FwuCyEorpRvB0_sA9r7d7J0yQjEU4s0Up8vj7UqW-8sTfDybr7eubKf7JhfDYE4KSxr_o_umVTE30JxqiDg6PRcpbWQZf1ZjHFuepNEf_As9iZFSs7vRA0su7AMw1vgEjyg-6XGQ7-WznYH2FIu0cFU7BzwnPxlY7YkHTEnekixtwl6I_c27i2Y_t89kQbUkUNH5g6FiHfdvs./download'
sudo apt-get install libomp-dev
pip3 install numpy==1.17.3
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl