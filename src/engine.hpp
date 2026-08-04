#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <vector>
#include <string>

namespace vision_engine {

enum class ShaderType {
    PBR_Standard = 0,
    Unlit,
    Transparent,
    Phong,
    Custom
};

struct Material {
    std::string name = "DefaultMaterial";
    ShaderType shader_type = ShaderType::PBR_Standard;
    float base_color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    std::string normal_map = "";
    float roughness = 0.5f;
    float metallic = 0.0f;
    float emission[3] = {0.0f, 0.0f, 0.0f};
    float opacity = 1.0f;
};

struct Entity {
    int id = 0;
    std::string name = "Entity";
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    float width = 1.0f;
    float height = 1.0f;
    float depth = 1.0f;
    bool active = true;
    Material material;
};

struct Block {
    float x;
    float y;
    float width;
    float height;
    float health;
    float max_health;
    bool active;
    Material material;
};

struct Debris {
    float x;
    float y;
    float vx;
    float vy;
    float width;
    float height;
    float lifespan;
    bool active;
    Material material;
};

class GameEngine {
public:
    GameEngine(float width = 800.0f, float height = 600.0f);

    void update(float dt);
    void set_target_position(float x, float y);
    
    // General Entity Management
    int add_entity(std::string name = "Entity", float x = 0.0f, float y = 0.0f, float z = 0.0f,
                   float vx = 0.0f, float vy = 0.0f, float vz = 0.0f,
                   float w = 1.0f, float h = 1.0f, float d = 1.0f,
                   Material mat = Material());
    const std::vector<Entity>& get_entities() const { return m_entities; }
    void clear_entities();

    // Legacy Block & Debris Management
    void add_block(float x, float y, float w, float h, float health, Material mat = Material());
    const std::vector<Block>& get_blocks() const { return m_blocks; }
    const std::vector<Debris>& get_debris() const { return m_debris; }
    void clear_blocks();

    float get_x() const { return m_x; }
    float get_y() const { return m_y; }
    float get_target_x() const { return m_target_x; }
    float get_target_y() const { return m_target_y; }
    float get_width() const { return m_width; }
    float get_height() const { return m_height; }

private:
    float m_width;
    float m_height;
    
    // Ball / Sprite position & physics
    float m_x;
    float m_y;
    float m_vx;
    float m_vy;
    float m_gravity;
    float m_radius;

    // AI Vision Target (e.g. detected face position)
    float m_target_x;
    float m_target_y;

    int m_next_entity_id = 1;
    std::vector<Entity> m_entities;
    std::vector<Block> m_blocks;
    std::vector<Debris> m_debris;
};

} // namespace vision_engine

#endif // ENGINE_HPP
