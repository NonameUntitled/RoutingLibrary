import argparse
import json
import xml.etree.ElementTree as ET
import re


# Open misc/example_with_instruction.xml file in draw.io (https://app.diagrams.net/)
# to see instructions on how to use it
def parse_draw_io_to_topology():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    tree = ET.parse(args.input)
    root = tree.getroot()

    source_id = 0
    sink_id = 0
    diverter_id = 0
    junction_id = 0
    id_map = {}
    nodes = []
    edges = []

    def node_by_unique_id(unique_id):
        for node in nodes:
            if node["unique_id"] == unique_id:
                return node
        return None

    for cell in root.findall(".//mxCell"):
        style = cell.get("style")
        if style and "triangle" in style and "direction=north" in style:
            geometry = cell.find("mxGeometry")
            if geometry is not None:
                nodes.append({
                    "type": "source",
                    "id": source_id,
                    "unique_id": cell.get("id"),
                    "x": float(geometry.get("x")),
                    "y": float(geometry.get("y")),
                })
                id_map[cell.get("id")] = source_id
                source_id += 1
                continue
            else:
                print("No geometry for source", cell.get("id"))
        if style and "triangle" in style and "direction=south" in style:
            geometry = cell.find("mxGeometry")
            if geometry is not None:
                nodes.append({
                    "type": "sink",
                    "id": sink_id,
                    "unique_id": cell.get("id"),
                    "x": float(geometry.get("x")),
                    "y": float(geometry.get("y")),
                })
                id_map[cell.get("id")] = sink_id
                sink_id += 1
                continue
            else:
                print("No geometry for sink", cell.get("id"))
        if style and "ellipse" in style:
            geometry = cell.find("mxGeometry")
            if geometry is not None:
                nodes.append({
                    "type": "diverter",
                    "id": diverter_id,
                    "unique_id": cell.get("id"),
                    "x": float(geometry.get("x")),
                    "y": float(geometry.get("y")),
                })
                id_map[cell.get("id")] = diverter_id
                diverter_id += 1
                continue
            else:
                print("No geometry for diverter", cell.get("id"))
        if style and "rhombus" in style:
            geometry = cell.find("mxGeometry")
            if geometry is not None:
                nodes.append({
                    "type": "junction",
                    "id": junction_id,
                    "unique_id": cell.get("id"),
                    "x": float(geometry.get("x")),
                    "y": float(geometry.get("y")),
                })
                id_map[cell.get("id")] = junction_id
                junction_id += 1
                continue
            else:
                print("No geometry for junction", cell.get("id"))

    for cell in root.findall(".//mxCell"):
        source = node_by_unique_id(cell.get("source"))
        target = node_by_unique_id(cell.get("target"))
        if source is None or target is None:
            continue
        value = cell.get("value")
        length = None
        quality = None
        if value is not None:
            pattern = r"\{\s*l:\s*([\d.]+),\s*q:\s*([\d.]+)\}"
            match = re.search(pattern, value)
            if match is not None:
                length = match.group(1)
                quality = match.group(2)
                if length is not None:
                    length = float(length)
                if quality is not None:
                    quality = float(quality)
        style = cell.get("style")
        color = None
        if style is not None:
            pattern = r"strokeColor=#(?:[0-9a-fA-F]{3}){1,2};"
            match = re.search(pattern, style)
            if match is not None:
                color = match.group(0)
                if color is not None:
                    color = color.split("=")[1]
                    if color is not None:
                        color = color[1:]
                        if len(color) == 3:
                            color = color[0] + color[0] + color[1] + color[1] + color[2] + color[2]
                        color = "#" + color
        if color is None:
            print("No color for conveyor", cell.get("id"))
        if source is not None and target is not None:
            edges.append({
                "source": source["unique_id"],
                "target": target["unique_id"],
                "length": length,
                "quality": quality,
                "color": color,
            })

    edge_dict = {}
    for edge in edges:
        edge_id = edge['color']
        if edge_id not in edge_dict:
            edge_dict[edge_id] = []
        edge_dict[edge_id].append(
            {'source': edge['source'], 'target': edge['target'], 'length': edge['length'], 'quality': edge['quality']})

    def sort_edges(edge_group):
        sorted_group = []

        outgoing = {edge['source']: edge for edge in edge_group}

        all_froms = {edge['source'] for edge in edge_group}
        all_tos = {edge['target'] for edge in edge_group}
        start = list(all_froms - all_tos)[0]

        while start in outgoing:
            edge = outgoing[start]
            sorted_group.append(edge)
            start = edge['target']

        return sorted_group

    conveyors = []
    for edge_color, edges in edge_dict.items():
        conveyors.append({'id': len(conveyors), 'color': edge_color, 'edges': sort_edges(edges)})

    sinks_config = []
    for node in nodes:
        if node['type'] == 'sink':
            sinks_config.append({"id": node['id'], "a_x": node['x'], "a_y": node['y']})

    def find_source_upstream_conv(source_unique_id):
        for conveyor in conveyors:
            first_edge_node = conveyor['edges'][0]['source']
            if first_edge_node == source_unique_id:
                return conveyor['id']
        return None

    sources_config = {}
    for node in nodes:
        if node['type'] == 'source':
            sources_config[str(node['id'])] = {"upstream_conv": find_source_upstream_conv(node['unique_id']),
                                               "a_x": node['x'], "a_y": node['y']}

    def find_diverter_info(diverter_unique_id):
        upstream_conv = None
        pos = None
        conveyor = None
        for conveyor in conveyors:
            first_edge_node = conveyor['edges'][0]['source']
            if first_edge_node == diverter_unique_id:
                upstream_conv = conveyor['id']
                break
        for conveyor in conveyors:
            length = 0
            is_found = False
            for edge in conveyor['edges']:
                length += edge['length']
                if edge['target'] == diverter_unique_id:
                    is_found = True
                    break
            if is_found:
                pos = length
                conveyor = conveyor['id']
                break
        return {"upstream_conv": upstream_conv, "pos": pos, "conveyor": conveyor}

    diverters_config = {}
    for node in nodes:
        if node['type'] == 'diverter':
            diverter_info = find_diverter_info(node['unique_id'])
            diverters_config[str(node['id'])] = {"upstream_conv": diverter_info["upstream_conv"],
                                                 "pos": diverter_info["pos"],
                                                 "conveyor": diverter_info["conveyor"],
                                                 "a_x": node['x'], "a_y": node['y']}

    def get_conveyor_upstream(last_target_unique_id):
        node = node_by_unique_id(last_target_unique_id)
        if node is None:
            return None
        if node['type'] == 'sink':
            return {"type": "sink",
                    "id": node['id']}
        if node['type'] == 'junction':
            upstream = None
            pos = None
            for conveyor in conveyors:
                last_edge_node = conveyor['edges'][-1]['target']
                if last_edge_node == last_target_unique_id:
                    continue
                length = 0
                is_found = False
                for edge in conveyor['edges']:
                    length += edge['length']
                    if edge['target'] == last_target_unique_id:
                        is_found = True
                        break
                if is_found:
                    pos = length
                    upstream = conveyor['id']
                    break
            return {"type": "conveyor",
                    "id": upstream,
                    "pos": pos,
                    "a_x": node['x'],
                    "a_y": node['y']}
        return None

    conveyors_config = {}
    for conveyor in conveyors:
        conveyors_config[str(conveyor['id'])] = {
            "length": sum([edge['length'] for edge in conveyor['edges']]),
            "upstream": get_conveyor_upstream(conveyor['edges'][-1]['target']),
            "quality": min([edge['quality'] for edge in conveyor['edges']])
        }

    with open(args.output, 'w') as outfile:
        json.dump({
            "topology": {
                "type": "oriented",
                "sinks": sinks_config,
                "sources": sources_config,
                "diverters": diverters_config,
                "conveyors": conveyors_config,
            }
        }, outfile, indent=2)


if __name__ == '__main__':
    parse_draw_io_to_topology()
