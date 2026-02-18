// Three.js 3D Scene for AgriFert Predict
// Creates an interactive 3D agricultural visualization

let scene, camera, renderer, soil, controls;
let animationId;

function initThreeScene() {
    // Get the container element
    const container = document.getElementById('agri-3d-scene');
    if (!container) return;

    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB); // Sky blue background

    // Camera setup
    camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);

    // Renderer setup
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Lighting
    setupLighting();

    // Create 3D objects
    createSoil();
    createGround();
    createPlants();
    createSkybox();

    // Controls
    setupControls();

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    // Start animation
    animate();
}

function setupLighting() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    // Directional light (sun)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;

    // Configure shadow properties
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 50;
    directionalLight.shadow.camera.left = -10;
    directionalLight.shadow.camera.right = 10;
    directionalLight.shadow.camera.top = 10;
    directionalLight.shadow.camera.bottom = -10;

    scene.add(directionalLight);

    // Hemisphere light for more natural lighting
    const hemisphereLight = new THREE.HemisphereLight(0x87CEEB, 0x8B4513, 0.3);
    scene.add(hemisphereLight);
}

function createSoil() {
    // Create soil cylinder
    const soilGeometry = new THREE.CylinderGeometry(3, 3, 1, 32);
    const soilMaterial = new THREE.MeshLambertMaterial({
        color: 0x8B4513, // Brown color
        roughness: 0.8,
        metalness: 0.1
    });

    soil = new THREE.Mesh(soilGeometry, soilMaterial);
    soil.position.y = -0.5;
    soil.castShadow = true;
    soil.receiveShadow = true;
    scene.add(soil);

    // Add soil texture details
    addSoilDetails();
}

function addSoilDetails() {
    // Add some small rocks/particles on the soil surface
    for (let i = 0; i < 20; i++) {
        const rockGeometry = new THREE.SphereGeometry(0.05 + Math.random() * 0.1, 8, 8);
        const rockMaterial = new THREE.MeshLambertMaterial({
            color: 0x654321 + Math.random() * 0x222222
        });

        const rock = new THREE.Mesh(rockGeometry, rockMaterial);
        rock.position.set(
            (Math.random() - 0.5) * 5,
            -0.3 + Math.random() * 0.2,
            (Math.random() - 0.5) * 5
        );
        rock.castShadow = true;
        scene.add(rock);
    }
}

function createGround() {
    // Create ground plane
    const groundGeometry = new THREE.PlaneGeometry(20, 20);
    const groundMaterial = new THREE.MeshLambertMaterial({
        color: 0x228B22, // Forest green
        transparent: true,
        opacity: 0.8
    });

    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1;
    ground.receiveShadow = true;
    scene.add(ground);
}

function createPlants() {
    // Create some simple plant representations
    const plantTypes = ['wheat', 'corn', 'rice'];
    const colors = [0xFFD700, 0x32CD32, 0x98FB98];

    for (let i = 0; i < 15; i++) {
        const plantType = Math.floor(Math.random() * plantTypes.length);
        const plantGeometry = createPlantGeometry(plantTypes[plantType]);
        const plantMaterial = new THREE.MeshLambertMaterial({
            color: colors[plantType],
            transparent: true,
            opacity: 0.8
        });

        const plant = new THREE.Mesh(plantGeometry, plantMaterial);

        // Random position around the soil
        const angle = (i / 15) * Math.PI * 2;
        const radius = 2 + Math.random() * 2;
        plant.position.set(
            Math.cos(angle) * radius,
            0,
            Math.sin(angle) * radius
        );

        plant.castShadow = true;
        scene.add(plant);

        // Add slight random rotation
        plant.rotation.y = Math.random() * Math.PI * 2;
    }
}

function createPlantGeometry(type) {
    switch (type) {
        case 'wheat':
            // Simple wheat stalk with head
            const wheatGeometry = new THREE.CylinderGeometry(0.02, 0.02, 1, 8);
            return wheatGeometry;

        case 'corn':
            // Corn stalk
            const cornGeometry = new THREE.CylinderGeometry(0.03, 0.03, 1.5, 8);
            return cornGeometry;

        case 'rice':
            // Rice plant (thinner)
            const riceGeometry = new THREE.CylinderGeometry(0.015, 0.015, 0.8, 6);
            return riceGeometry;

        default:
            return new THREE.CylinderGeometry(0.02, 0.02, 1, 8);
    }
}

function createSkybox() {
    // Create a simple sky gradient effect
    const skyGeometry = new THREE.SphereGeometry(100, 32, 32);
    const skyMaterial = new THREE.ShaderMaterial({
        vertexShader: `
            varying vec3 vWorldPosition;
            void main() {
                vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                vWorldPosition = worldPosition.xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            varying vec3 vWorldPosition;
            void main() {
                float h = normalize(vWorldPosition).y;
                gl_FragColor = vec4(mix(vec3(0.5, 0.7, 1.0), vec3(0.0, 0.4, 0.8), max(pow(max(h, 0.0), 0.8), 0.0)), 1.0);
            }
        `,
        side: THREE.BackSide
    });

    const sky = new THREE.Mesh(skyGeometry, skyMaterial);
    scene.add(sky);
}

function setupControls() {
    // Auto-rotation camera orbit (no OrbitControls add-on needed)
    // Camera orbits around the scene automatically
}

function onWindowResize() {
    const container = document.getElementById('agri-3d-scene');
    if (!container) return;

    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

let cameraAngle = Math.PI / 4;

function animate() {
    animationId = requestAnimationFrame(animate);

    // Rotate soil slowly
    if (soil) {
        soil.rotation.y += 0.005;
    }

    // Auto-orbit camera around the scene
    cameraAngle += 0.003;
    const radius = 7;
    camera.position.x = Math.cos(cameraAngle) * radius;
    camera.position.z = Math.sin(cameraAngle) * radius;
    camera.position.y = 4 + Math.sin(cameraAngle * 0.5) * 1;
    camera.lookAt(0, 0, 0);

    // Render scene
    renderer.render(scene, camera);
}

function cleanupThreeScene() {
    if (animationId) {
        cancelAnimationFrame(animationId);
    }

    if (renderer) {
        renderer.dispose();
    }

    // Remove event listeners
    window.removeEventListener('resize', onWindowResize);
}

// Initialize scene when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    // Check if Three.js is loaded
    if (typeof THREE !== 'undefined') {
        initThreeScene();
    } else {
        console.warn('Three.js not loaded. 3D scene will not be available.');
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', cleanupThreeScene);

// Export functions for potential external use
window.AgriFert3D = {
    init: initThreeScene,
    cleanup: cleanupThreeScene,
    scene: () => scene,
    camera: () => camera,
    renderer: () => renderer
};
