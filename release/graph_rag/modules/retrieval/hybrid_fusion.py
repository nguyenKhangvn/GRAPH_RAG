# graph_rag/modules/retrieval/fusion.py
from typing import List, Dict

def reciprocal_rank_fusion(vector_results: List[Dict], keyword_results: List[Dict], k=60) -> List[Dict]:
    """
    Thuật toán RRF (Reciprocal Rank Fusion) để gộp kết quả từ Vector và Keyword.
    """
    fused_scores = {}
    
    VECTOR_WEIGHT = 1.0 
    KEYWORD_WEIGHT = 1.5 

    def add_scores(item_list, weight, source_type_override=None):
        for rank, item in enumerate(item_list):
            node_id = item['id']
            
            if node_id not in fused_scores:
                fused_scores[node_id] = {
                    'data': item, 
                    'score': 0.0, 
                    'sources': set()
                }
            
            # Công thức RRF chuẩn: 1 / (k + rank)
            # Rank bắt đầu từ 0 nên cộng thêm 1
            rank_score = 1 / (k + rank + 1)
            
            # Cộng dồn điểm
            fused_scores[node_id]['score'] += weight * rank_score
            
            # Ghi nhận nguồn tìm thấy (Vector hay Fulltext)
            # Lấy từ dữ liệu gốc 'found_by', nếu không có thì dùng override
            src = item.get('found_by', source_type_override)
            fused_scores[node_id]['sources'].add(src)

    # 1. Cộng điểm từ Vector Search
    # Lưu ý: vector_results thường chứa các dict có key 'found_by': 'vector'
    add_scores(vector_results, VECTOR_WEIGHT, source_type_override="vector")
    
    # 2. Cộng điểm từ Fulltext Search
    add_scores(keyword_results, KEYWORD_WEIGHT, source_type_override="fulltext")
    
    # 3. Sắp xếp kết quả cuối cùng (Score giảm dần)
    sorted_results = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
    
    final_list = []
    for item in sorted_results:
        data = item['data']
        # Làm tròn điểm số cho đẹp
        data['final_score'] = round(item['score'], 6)
        # Chuyển set sources thành list để dễ xử lý về sau
        data['found_by'] = list(item['sources'])
        final_list.append(data)
        
    return final_list