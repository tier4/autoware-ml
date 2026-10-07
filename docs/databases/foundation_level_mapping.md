# Foundation level mapping

The `foundation` taxonomy level trains the offline foundation model (see
[Offline foundation model](../user-guide/foundation_model.md)). It keeps the 33 classes of the
fine level, appends `traffic_sign` and `traffic_light` (35 segmentation classes), keeps the 20
detection classes of the fine level, and adds *label sets*: targets that name a group of
classes when a corpus cannot tell the classes apart. A set point trains the summed probability
of its classes (see [Database design](design.md)) and is not scored.

Every corpus reaches the level through the foundation vocabularies
(`taxonomy/vocabulary/foundation_*.yaml`, the default vocabularies plus a few names). A corpus
whose raw names are coarser than their spelling suggests declares `category_aliases` in its
scenario configuration, so the same raw name can mean a class in one corpus and a set in
another:

1. **J6 gen2 semantic segmentation v1** (ground truth at the old J6 gen2 specification, stages
   1 and 2): `manmade` covers buildings, poles, signs, lights and unboxed barriers, and the
   curbs are folded into `sidewalk` and `other_flat_surface`.
2. **10 Hz pseudo labels** (stages 2 and 3): predicted by the 26 class old specification model
   and written with L1 names, so they have the old granularity: `manmade.building` is the old
   `manmade`, the non-driveable flat names contain the curbs, `noise.noise` the ghost points.
   The category tables also list the new L1 names, but no pseudo point carries
   `vertical_thin`, a curb, `manmade.solid_stuff` or `noise.ghost_point`.
3. **Kognic** (new specification, stage 3): the only corpus with class level supervision for
   `vertical_thin`, the curbs, `manmade.solid_stuff`, `noise.ghost_point`, `ego_vehicle`,
   `traffic_sign` and `traffic_light`.

The tables are generated from the category tables of the corpora and the configuration.

## Label sets

| set | member classes |
|---|---|
| `thin_or_sign` | `vertical_thin`, `traffic_sign`, `traffic_light` |
| `manmade_or_thin` | `manmade.building`, `manmade.solid_stuff`, `manmade.special`, `vertical_thin`, `traffic_sign`, `traffic_light`, `barrier.movable`, `barrier.construction_sign` |
| `sidewalk_or_curb` | `non_driveable_flat.sidewalk`, `curb.low_curb`, `curb.high_curb` |
| `other_flat_or_curb` | `non_driveable_flat.other_flat_surface`, `curb.low_curb`, `curb.high_curb` |
| `any_noise` | `noise.noise`, `noise.ghost_point` |

## J6 gen2 semantic segmentation v1

| raw category | foundation target |
|---|---|
| `animal` | `animal` |
| `bicycle` | `bicycle` |
| `bus` | `bus` |
| `car` | `car.common` |
| `construction_vehicle` | `truck.construction` |
| `debris` | `debris.unclassified` |
| `drivable_surface` | `driveable_flat` |
| `emergency_vehicle` | `car.emergency` |
| `forklift` | `truck.construction` |
| `ghost_point` | `noise.ghost_point` |
| `kart` | `car.special` |
| `manmade` | set `manmade_or_thin` (via `legacy_manmade`) |
| `motorcycle` | `motorcycle` |
| `noise` | `noise.noise` |
| `other_flat_surface` | set `other_flat_or_curb` (via `legacy_other_flat_surface`) |
| `other_stuff` | `manmade.special` |
| `out_of_sync` | ignored |
| `pedestrian` | `pedestrian.common` |
| `personal_mobility` | `pedestrian.personal_mobility` |
| `pushable_pullable` | `debris.pushable_pullable` |
| `semi_trailer` | `truck.trailer` |
| `sidewalk` | set `sidewalk_or_curb` (via `legacy_sidewalk`) |
| `stroller` | `pedestrian.stroller` |
| `tractor_unit` | `truck.common` |
| `traffic_cone` | `traffic_cone` |
| `train` | `train` |
| `truck` | `truck.common` |
| `unpainted` | ignored |
| `vegetation` | `vegetation` |

## 10 Hz pseudo labels

| raw category | foundation target |
|---|---|
| `animal` | `animal` |
| `barrier.construction_sign` | `barrier.construction_sign` |
| `barrier.movable` | `barrier.movable` |
| `bicycle` | `bicycle` |
| `bus` | `bus` |
| `car.common` | `car.common` |
| `car.emergency` | `car.emergency` |
| `car.special` | `car.special` |
| `curb.high_curb` | `curb.high_curb` |
| `curb.low_curb` | `curb.low_curb` |
| `debris.pushable_pullable` | `debris.pushable_pullable` |
| `debris.unclassified` | `debris.unclassified` |
| `driveable_flat` | `driveable_flat` |
| `ego_vehicle` | `ego_vehicle` |
| `ignored` | ignored |
| `manmade.building` | set `manmade_or_thin` (via `legacy_manmade`) |
| `manmade.solid_stuff` | `manmade.solid_stuff` |
| `manmade.special` | `manmade.special` |
| `motorcycle` | `motorcycle` |
| `noise.ghost_point` | `noise.ghost_point` |
| `noise.noise` | set `any_noise` (via `legacy_noise`) |
| `non_driveable_flat.other_flat_surface` | set `other_flat_or_curb` (via `legacy_other_flat_surface`) |
| `non_driveable_flat.sidewalk` | set `sidewalk_or_curb` (via `legacy_sidewalk`) |
| `pedestrian.common` | `pedestrian.common` |
| `pedestrian.personal_mobility` | `pedestrian.personal_mobility` |
| `pedestrian.stroller` | `pedestrian.stroller` |
| `traffic_cone` | `traffic_cone` |
| `train` | `train` |
| `truck.common` | `truck.common` |
| `truck.construction` | `truck.construction` |
| `truck.emergency` | `truck.emergency` |
| `truck.trailer` | `truck.trailer` |
| `vegetation` | `vegetation` |
| `vertical_thin` | set `thin_or_sign` |

## Kognic (new specification)

| raw category | foundation target |
|---|---|
| `ambulance` | `car.emergency` |
| `animal` | `animal` |
| `background` | ignored |
| `barrier` | `barrier.movable` |
| `bicycle` | `bicycle` |
| `bicycle_rack` | `manmade.special` |
| `bollard` | `vertical_thin` |
| `building` | `manmade.building` |
| `bus` | `bus` |
| `car` | `car.common` |
| `construction_sign` | `barrier.construction_sign` |
| `construction_vehicle` | `truck.construction` |
| `debris` | `debris.unclassified` |
| `drainage` | `driveable_flat` |
| `drivable_surface` | `driveable_flat` |
| `ego_vehicle` | `ego_vehicle` |
| `emergency_vehicle` | `car.emergency` |
| `fire_truck` | `truck.emergency` |
| `flag` | `vertical_thin` |
| `forklift` | `truck.construction` |
| `ghost_point` | `noise.ghost_point` |
| `high_curb` | `curb.high_curb` |
| `kart` | `car.special` |
| `low_curb` | `curb.low_curb` |
| `motorcycle` | `motorcycle` |
| `noise` | `noise.noise` |
| `open_door` | ignored |
| `other_flat_surface` | `non_driveable_flat.other_flat_surface` |
| `other_stuff` | `manmade.special` |
| `other_vehicle` | `car.special` |
| `out_of_sync` | ignored |
| `pedestrian` | `pedestrian.common` |
| `personal_mobility` | `pedestrian.personal_mobility` |
| `pole` | `vertical_thin` |
| `police_car` | `car.emergency` |
| `pushable_pullable` | `debris.pushable_pullable` |
| `sidewalk` | `non_driveable_flat.sidewalk` |
| `solid_stuff` | `manmade.solid_stuff` |
| `stroller` | `pedestrian.stroller` |
| `traffic_cone` | `traffic_cone` |
| `traffic_light` | `traffic_light` |
| `traffic_sign` | `traffic_sign` |
| `trailer` | `truck.trailer` |
| `train` | `train` |
| `truck` | `truck.common` |
| `vegetation` | `vegetation` |
| `vehicle_protruding_object` | ignored |

## Detection classes

20 box classes, the classes of the fine level: `car.common`, `car.emergency`, `car.special`,
`truck.common`, `truck.emergency`, `truck.trailer`, `truck.construction`, `bus`, `train`,
`motorcycle`, `bicycle`, `pedestrian.common`, `pedestrian.personal_mobility`,
`pedestrian.stroller`, `animal`, `barrier.movable`, `barrier.construction_sign`,
`traffic_cone`, `debris.unclassified`, `debris.pushable_pullable`. Box names of the foundation
corpora that are segmentation classes (L1 surface and structure names in the pseudo boxes,
Kognic cuboids of segmentation classes) are outside every detection level.
